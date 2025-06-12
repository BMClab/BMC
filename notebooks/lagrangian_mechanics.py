import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Lagrangian mechanics

        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](http://pesquisa.ufabc.edu.br/bmclab)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <center><div style="background-color:#f2f2f2;border:1px solid black;width:72%;padding:5px 10px 5px 10px;text-align:left;">
        <i>"The theoretical development of the laws of motion of bodies is a problem of such interest and importance, that it has engaged the attention of all the most eminent mathematicians, since the invention of dynamics as a mathematical science by <b>Galileo</b>, and especially since the wonderful extension which was given to that science by <b>Newton</b>. Among the successors of those illustrious men, <b>Lagrange</b> has perhaps done more than any other analyst, to give extent and harmony to such deductive researches, by showing that the most varied consequences respecting the motions of systems of bodies may be derived from one radical formula; the beauty of the method so suiting the dignity of the results, as to make of his great work a kind of scientific poem."</i> &nbsp; <b>Hamilton</b> (1834)
        </div></center>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Generalized-coordinates" data-toc-modified-id="Generalized-coordinates-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Generalized coordinates</a></span></li><li><span><a href="#Euler–Lagrange-equations" data-toc-modified-id="Euler–Lagrange-equations-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Euler–Lagrange equations</a></span><ul class="toc-item"><li><span><a href="#Steps-to-deduce-the-Euler-Lagrange-equations" data-toc-modified-id="Steps-to-deduce-the-Euler-Lagrange-equations-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Steps to deduce the Euler-Lagrange equations</a></span></li><li><span><a href="#Example:-Particle-moving-under-the-influence-of-a-conservative-force" data-toc-modified-id="Example:-Particle-moving-under-the-influence-of-a-conservative-force-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Example: Particle moving under the influence of a conservative force</a></span></li><li><span><a href="#Example:-Ideal-mass-spring-system" data-toc-modified-id="Example:-Ideal-mass-spring-system-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Example: Ideal mass-spring system</a></span></li><li><span><a href="#Example:-Simple-pendulum-under-the-influence-of-gravity" data-toc-modified-id="Example:-Simple-pendulum-under-the-influence-of-gravity-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Example: Simple pendulum under the influence of gravity</a></span><ul class="toc-item"><li><span><a href="#Numerical-solution-of-the-equation-of-motion-for-the-simple-pendulum" data-toc-modified-id="Numerical-solution-of-the-equation-of-motion-for-the-simple-pendulum-3.4.1"><span class="toc-item-num">3.4.1&nbsp;&nbsp;</span>Numerical solution of the equation of motion for the simple pendulum</a></span></li></ul></li><li><span><a href="#Python-code-to-automate-the-calculation-of-the-Euler–Lagrange-equation" data-toc-modified-id="Python-code-to-automate-the-calculation-of-the-Euler–Lagrange-equation-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Python code to automate the calculation of the Euler–Lagrange equation</a></span></li><li><span><a href="#Example:-Double-pendulum-under-the-influence-of-gravity" data-toc-modified-id="Example:-Double-pendulum-under-the-influence-of-gravity-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Example: Double pendulum under the influence of gravity</a></span><ul class="toc-item"><li><span><a href="#Numerical-solution-of-the-equation-of-motion-for-the-double-pendulum" data-toc-modified-id="Numerical-solution-of-the-equation-of-motion-for-the-double-pendulum-3.6.1"><span class="toc-item-num">3.6.1&nbsp;&nbsp;</span>Numerical solution of the equation of motion for the double pendulum</a></span></li></ul></li><li><span><a href="#Example:-Double-compound-pendulum-under-the-influence-of-gravity" data-toc-modified-id="Example:-Double-compound-pendulum-under-the-influence-of-gravity-3.7"><span class="toc-item-num">3.7&nbsp;&nbsp;</span>Example: Double compound pendulum under the influence of gravity</a></span></li><li><span><a href="#Example:-Double-compound-pendulum-in-joint-space" data-toc-modified-id="Example:-Double-compound-pendulum-in-joint-space-3.8"><span class="toc-item-num">3.8&nbsp;&nbsp;</span>Example: Double compound pendulum in joint space</a></span></li><li><span><a href="#Example:-Mass-attached-to-a-spring-on-a-horizontal-plane" data-toc-modified-id="Example:-Mass-attached-to-a-spring-on-a-horizontal-plane-3.9"><span class="toc-item-num">3.9&nbsp;&nbsp;</span>Example: Mass attached to a spring on a horizontal plane</a></span></li></ul></li><li><span><a href="#Generalized-forces" data-toc-modified-id="Generalized-forces-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Generalized forces</a></span><ul class="toc-item"><li><span><a href="#Example:-Simple-pendulum-on-moving-cart" data-toc-modified-id="Example:-Simple-pendulum-on-moving-cart-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Example: Simple pendulum on moving cart</a></span></li><li><span><a href="#Example:-Two-masses-and-two-springs-under-the-influence-of-gravity" data-toc-modified-id="Example:-Two-masses-and-two-springs-under-the-influence-of-gravity-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Example: Two masses and two springs under the influence of gravity</a></span></li><li><span><a href="#Example:-Mass-spring-damper-system-with-gravity" data-toc-modified-id="Example:-Mass-spring-damper-system-with-gravity-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Example: Mass-spring-damper system with gravity</a></span><ul class="toc-item"><li><span><a href="#Numerical-solution-of-the-equation-of-motion-for-mass-spring-damper-system" data-toc-modified-id="Numerical-solution-of-the-equation-of-motion-for-mass-spring-damper-system-4.3.1"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>Numerical solution of the equation of motion for mass-spring-damper system</a></span></li></ul></li><li><span><a href="#Example:-Mass-spring-damper-system-in-a-ramp-with-gravity" data-toc-modified-id="Example:-Mass-spring-damper-system-in-a-ramp-with-gravity-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Example: Mass-spring-damper system in a ramp with gravity</a></span><ul class="toc-item"><li><span><a href="#Numerical-solution-of-the-equation-of-motion-for-mass-spring-damper-system-in-a-ramp-with-gravity" data-toc-modified-id="Numerical-solution-of-the-equation-of-motion-for-mass-spring-damper-system-in-a-ramp-with-gravity-4.4.1"><span class="toc-item-num">4.4.1&nbsp;&nbsp;</span>Numerical solution of the equation of motion for mass-spring-damper system in a ramp with gravity</a></span></li></ul></li></ul></li><li><span><a href="#Forces-of-constraint" data-toc-modified-id="Forces-of-constraint-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Forces of constraint</a></span><ul class="toc-item"><li><span><a href="#Example:-Force-of-constraint-in-a-simple-pendulum-under-the-influence-of-gravity" data-toc-modified-id="Example:-Force-of-constraint-in-a-simple-pendulum-under-the-influence-of-gravity-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Example: Force of constraint in a simple pendulum under the influence of gravity</a></span></li></ul></li><li><span><a href="#Lagrangian-formalism-applied-to-non-mechanical-systems" data-toc-modified-id="Lagrangian-formalism-applied-to-non-mechanical-systems-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Lagrangian formalism applied to non-mechanical systems</a></span><ul class="toc-item"><li><span><a href="#Example:-Lagrangian-formalism-for-RLC-eletrical-circuits" data-toc-modified-id="Example:-Lagrangian-formalism-for-RLC-eletrical-circuits-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Example: Lagrangian formalism for RLC eletrical circuits</a></span></li></ul></li><li><span><a href="#Considerations-on-the-Lagrangian-mechanics" data-toc-modified-id="Considerations-on-the-Lagrangian-mechanics-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Considerations on the Lagrangian mechanics</a></span></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-internet" data-toc-modified-id="Video-lectures-on-the-internet-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Video lectures on the internet</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell
def _():
    # import necessary libraries and configure environment
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('notebook', font_scale=1.2, rc={"lines.linewidth": 2})
    # import Sympy functions
    import sympy as sym
    from sympy import Symbol, symbols, cos, sin, Matrix, simplify, Eq, latex, expand
    from sympy.solvers.solveset import nonlinsolve
    from sympy.physics.mechanics import dynamicsymbols, mlatex, init_vprinting
    init_vprinting()
    from IPython.display import display, Math
    return (
        Eq,
        Math,
        Symbol,
        cos,
        display,
        dynamicsymbols,
        latex,
        mlatex,
        nonlinsolve,
        np,
        plt,
        simplify,
        sin,
        sym,
        symbols,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction

        We know that some problems in dynamics can be solved using the principle of conservation of mechanical energy, that the total mechanical energy in a system (the sum of potential and kinetic energies) is constant when only conservative forces are present in the system. Such approach is one kind of energy methods, see for example, pages 495-512 in Ruina and Pratap (2019).  

        Lagrangian mechanics (after [Joseph-Louis Lagrange](https://en.wikipedia.org/wiki/Joseph-Louis_Lagrange)) can be seen as another kind of energy methods, but much more general, to the extent is an alternative to Newtonian mechanics.  

        The Lagrangian mechanics is a formulation of classical mechanics where the equations of motion are obtained from the kinetic and potential energy of the system (scalar quantities) represented in generalized coordinates instead of using Newton's laws of motion to deduce the equations of motion from the forces on the system (vector quantities) represented in Cartesian coordinates.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Generalized coordinates

        The direct application of Newton's laws to mechanical systems results in a set of equations of motion in terms of Cartesian coordinates of each of the particles that make up the system. In many cases, this is not the most convenient coordinate system to solve the problem or describe the movement of the system. For example, for a serial chain of rigid links, such as a member of the human body or from a robot manipulator, it may be simpler to describe the positions of each link by the angles between links.  

        Coordinate systems such as angles of a chain of links are referred as [generalized coordinates](https://en.wikipedia.org/wiki/Generalized_coordinates). Generalized coordinates uniquely specify the positions of the particles in a system. Although there may be several generalized coordinates to describe a system, usually a judicious choice of generalized coordinates provides the minimum number of independent coordinates that define the configuration of a system (which is the number of <a href="https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics)">degrees of freedom</a> of the system), turning the problem simpler to solve. In this case, when the number of generalized coordinates equals the number of degrees of freedom, the system is referred as a holonomic system. In a non-holonomic system, the number of generalized coordinates necessary do describe the system depends on the path taken by the system.

        Being a little more technical, according to [Wikipedia](https://en.wikipedia.org/wiki/Configuration_space_(physics)):  
        "In classical mechanics, the parameters that define the configuration of a system are called generalized coordinates, and the vector space defined by these coordinates is called the configuration space of the physical system. It is often the case that these parameters satisfy mathematical constraints, such that the set of actual configurations of the system is a manifold in the space of generalized coordinates. This manifold is called the configuration manifold of the system."

        In problems where it is desired to use generalized coordinates, one can write Newton's equations of motion in terms of Cartesian coordinates and then transform them into generalized coordinates. However, it would be desirable and convenient to have a general method that would directly establish the equations of motion in terms of a set of convenient generalized coordinates. In addition, general methods for writing, and perhaps solving, the equations of motion in terms of any coordinate system would also be desirable. The [Lagrangian mechanics](https://en.wikipedia.org/wiki/Lagrangian_mechanics) is such a method.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Euler–Lagrange equations


        See [this notebook](http://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/lagrangian_mechanics_generalized.ipynb) for a deduction of the Lagrange's equation in generalized coordinates.

        Consider a system whose configuration (positions) can be described by a set of$N$generalized coordinates$q_i\,(i=1,\dotsc,N)$.

        Let's define the Lagrange or Lagrangian function$\mathcal{L}$as the difference between the total kinetic energy$T$and the total potential energy$V$of the system in terms of the generalized coordinates as:
        <p>
        <span class="notranslate">$\mathcal{L}(t,q,\dot{q}) = T(\dot{q}_1(t),\dotsc,\dot{q}_N(t)) - V(q_1(t),\dotsc,q_N(t))$</span>
    
        where the total potential energy is only due to conservative forces, that is, forces in which the total work done to move the system between two points is independent of the path taken.
    
        The Euler–Lagrange equations (or Lagrange's equations of the second kind) of the system are (omitting the functions' dependencies for sake of clarity):
        <p>
        <span class="notranslate">$\frac{\mathrm d }{\mathrm d t}\left( {\frac{\partial \mathcal{L}}{\partial \dot{q}_i }}
        \right)-\frac{\partial \mathcal{L}}{\partial q_i } = Q_{NCi} \quad i=1,\dotsc,N$</span>    
    
        where$Q_{NCi}$are the generalized forces due to non-conservative forces acting on the system, any forces that can't be expressed in terms of a potential.  

        Once all derivatives of the Lagrangian function are calculated and substitute them in the equations above, the result is the equation of motion (EOM) for each generalized coordinate. There will be$N$equations for a system with$N$generalized coordinates.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Steps to deduce the Euler-Lagrange equations

        1. Model the problem. Define the number of degrees of freedom. Carefully select the corresponding generalized coordinates to describe the system;
        2. Calculate the total kinetic and total potential energies of the system. Calculate the Lagrangian;
        3. Calculate the generalized forces for each generalized coordinate;
        4. For each generalized coordinate, calculate the three derivatives present on the left side of the Euler-Lagrange equation;
        5. For each generalized coordinate, substitute the result of these three derivatives in the left side and the corresponding generalized forces in the right side of the Euler-Lagrange equation.

        The EOM's, one for each generalized coordinate, are the result of the last step.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Particle moving under the influence of a conservative force

        Let's deduce the EOM of a particle with mass$m$moving in the three-dimensional space under the influence of a [conservative force](https://en.wikipedia.org/wiki/Conservative_force).  

        The model is the particle moving in 3D space and there is no generalized force (non-conservative force); the particle has three degrees of freedom and we need three generalized coordinates, which can be$(x, y, z)$, where$y$is vertical, in a Cartesian frame of reference.  
        The Lagrangian$(\mathcal{L} = T - V)$of the particle is:
        <p>
        <span class="notranslate">$\mathcal{L} = \frac{1}{2}m(\dot x^2(t) + \dot y^2(t) + \dot z^2(t)) - V(x(t),y(t),z(t))$</span>

        The equations of motion for the particle are found by applying the Euler–Lagrange equation for each coordinate.  
        For the$x$coordinate:
        <p>
        <span class="notranslate">$\frac{\mathrm d }{\mathrm d t}\left( {\frac{\partial \mathcal{L}}{\partial \dot{x}}}
        \right) - \frac{\partial \mathcal{L}}{\partial x } = 0$</span>

        And the derivatives are:
        <p>
        <span class="notranslate">$\begin{array}{rcl}
        &\dfrac{\partial \mathcal{L}}{\partial x} &=& -\dfrac{\partial V}{\partial x} \\
        &\dfrac{\partial \mathcal{L}}{\partial \dot{x}} &=& m\dot{x} \\
        &\dfrac{\mathrm d }{\mathrm d t}\left( {\dfrac{\partial \mathcal{L}}{\partial \dot{x}}} \right) &=& m\ddot{x}
        \end{array}$</span>

        Finally, the EOM is:
        <p>
        <span class="notranslate">$\begin{array}{l}
        m\ddot{x} + \dfrac{\partial V}{\partial x} = 0 \quad \rightarrow \\
        m\ddot{x} = -\dfrac{\partial V}{\partial x}
        \end{array}$</span>

        and same procedure for the$y$and$z$coordinates.  

        The equation above is the Newton's second law of motion.

        For instance, if the conservative force is due to the gravitational field near Earth's surface$(V=[0, mgy, 0])$, the Euler–Lagrange equations (the EOM's) are:
        <p>
        <span class="notranslate">$\begin{array}{rcl}
        m\ddot{x} &=& -\dfrac{\partial (0)}{\partial x} &=& 0 \\
        m\ddot{y} &=& -\dfrac{\partial (mgy)}{\partial y} &=& -mg \\
        m\ddot{z} &=& -\dfrac{\partial (0)}{\partial z} &=& 0
        \end{array}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Ideal mass-spring system

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/massspring_lagrange.png?raw=1" width="220" alt="mass spring" style="float:right;margin: 0px 20px 10px 20px;"/></figure>

        Consider a system with a mass$m$attached to an ideal spring (massless, length$\ell_0$, and spring constant$k$) at the horizontal direction$x$. A force is momentarily applied to the mass and then the system is left unperturbed.  
        Let's deduce the EOM of this system.  

        The system can be modeled as a particle attached to a spring moving at the direction$x$, the only generalized coordinate needed (with origin of the Cartesian reference frame at the wall where the spring is attached), and there is no generalized force.  
        The Lagrangian$(\mathcal{L} = T - V)$of the system is:
        <p>
        <span class="notranslate">$\mathcal{L} = \frac{1}{2}m\dot x^2 - \frac{1}{2}k(x-\ell_0)^2$</span>

        And the derivatives are:
        <p>
        <span class="notranslate">$\begin{array}{rcl}
        &\dfrac{\partial \mathcal{L}}{\partial x} &=& -k(x-\ell_0) \\
        &\dfrac{\partial \mathcal{L}}{\partial \dot{x}} &=& m\dot{x} \\
        &\dfrac{\mathrm d }{\mathrm d t}\left( {\dfrac{\partial \mathcal{L}}{\partial \dot{x}}} \right) &=& m\ddot{x}
        \end{array}$</span>

        Finally, the Euler–Lagrange equation (the EOM) is:
        <p>
        <span class="notranslate">$m\ddot{x} + k(x-\ell_0) = 0$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Simple pendulum under the influence of gravity

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/simplependulum_lagrange.png?raw=1" width="220" alt="simple pendulum" style="float:right;margin: 10px 50px 10px 50px;"/></figure>

        Consider a pendulum with a massless rod of length$d$and a mass$m$at the extremity swinging in a plane forming the angle$\theta$with the vertical.  
        Let's deduce the EOM of this system.

        The model is a particle oscillating as a pendulum under a constant gravitational force$-mg$.  
        Although the pendulum moves at the plane, it only has one degree of freedom, which can be described by the angle$\theta$, the generalized coordinate. Let's adopt the origin of the reference frame at the point of the pendulum suspension.  

        The kinetic energy of the system is:
        <p>
        <span class="notranslate">$T = \frac{1}{2}mv^2 = \frac{1}{2}m(\dot{x}^2+\dot{y}^2)$</span>

        where$\dot{x}$and$\dot{y}$are:
        <p>
        <span class="notranslate">$\begin{array}{l}
        x = d\sin(\theta) \\
        y = -d\cos(\theta) \\    
        \dot{x} = d\cos(\theta)\dot{\theta} \\
        \dot{y} = d\sin(\theta)\dot{\theta}
        \end{array}$</span>

        Consequently, the kinetic energy is:
        <p>
        <span class="notranslate">$T = \frac{1}{2}m\left((d\cos(\theta)\dot{\theta})^2 + (d\sin(\theta)\dot{\theta})^2\right) = \frac{1}{2}md^2\dot{\theta}^2$</span>    

        And the potential energy of the system is:
        <p>
        <span class="notranslate">$V = -mgy = -mgd\cos\theta$</span>
    
        The Lagrangian function is:
        <p>
        <span class="notranslate">$\mathcal{L} = \frac{1}{2}md^2\dot\theta^2 + mgd\cos\theta$</span>
    
        And the derivatives are:
        <p>
        <span class="notranslate">$\begin{array}{rcl}
        &\dfrac{\partial \mathcal{L}}{\partial \theta} &=& -mgd\sin\theta \\
        &\dfrac{\partial \mathcal{L}}{\partial \dot{\theta}} &=& md^2\dot{\theta} \\
        &\dfrac{\mathrm d }{\mathrm d t}\left( {\dfrac{\partial \mathcal{L}}{\partial \dot{\theta}}} \right) &=& md^2\ddot{\theta}
        \end{array}$</span>
    
        Finally, the Euler–Lagrange equation (the EOM) is:
        <p>
        <span class="notranslate">$md^2\ddot\theta + mgd\sin\theta = 0$</span>
    
        Note that although the generalized coordinate of the system is$\theta$, we had to employ Cartesian coordinates at the beginning to derive expressions for the kinetic and potential energies. For kinetic energy, we could have used its equivalent definition for circular motion$(T=I\dot{\theta}^2/2=md^2\dot{\theta}^2/2)$, but for the potential energy there is no other way since the gravitational force acts in the vertical direction.
        In cases like this, a fundamental aspect is to express the Cartesian coordinates in terms of the generalized coordinates.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Numerical solution of the equation of motion for the simple pendulum

        A classical approach to solve analytically the EOM for the simple pendulum is to consider the motion for small angles where$\sin\theta \approx \theta$and the differential equation is linearized to$d\ddot\theta + g\theta = 0$. This equation has an analytical solution of the type$\theta(t) = A \sin(\omega t + \phi)$, where$\omega = \sqrt{g/d}$and$A$and$\phi$are constants related to the initial position and velocity.  
        For didactic purposes, let's solve numerically the differential equation for the pendulum using [Euler’s method](https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/OrdinaryDifferentialEquation.ipynb#Euler-method).  

        Remember that we have to:  
        1. Transform the second-order ODE into two coupled first-order ODEs,  
        2. Approximate the derivative of each variable by its discrete first order difference  
        3. Write an equation to calculate the variable in a recursive way, updating its value with an equation based on the first order difference.  

        We will also implement different variations of the Euler method: Forward (standard), Semi-implicit, and Semi-implicit variation (same results as Semi-implicit).

        Implementing these steps in Python:  
        """
    )
    return


@app.cell
def _(np, plt):
    def euler_method(T=10, y0=[0, 0], h=0.01, method=2):
        """
        First-order numerical approximation for solving two coupled first-order ODEs.

        A first-order differential equation is an initial value problem of the form:
        y'(t) = f(t, y(t))  ;  y(t0) = y0

        Parameters:
            T: total period (in s) of the numerical integration
            y0: initial state [position, velocity]
            h: step for the numerical integration
            method: Euler method implementation, one of the following:
                1: 'forward' (standard)
                2: 'semi-implicit' (a.k.a., symplectic, Euler–Cromer)
                3: 'semi-implicit variation' (same results as 'semi-implicit')
        Two coupled first-order ODEs:
            dydt = v
            dvdt = a  # calculate the expression for acceleration at each step
        Two equations to update the values of the variables based on first-order difference:
            y[i+1] = y[i] + h*v[i]
            v[i+1] = v[i] + h*dvdt[i]
        Returns arrays time, [position, velocity]
        """
        N = int(np.ceil(T / _h))
        y = np.zeros((2, N))
        y[:, 0] = _y0
        t = np.linspace(0, T, N, endpoint=False)
        for i in range(N - 1):
            if method == 1:
                y[0, i + 1] = y[0, i] + _h * y[1, i]
                y[1, i + 1] = y[1, i] + _h * _dvdt(t[i], y[:, i])
            elif method == 2:
                y[1, i + 1] = y[1, i] + _h * _dvdt(t[i], y[:, i])
                y[0, i + 1] = y[0, i] + _h * y[1, i + 1]
            elif method == 3:
                y[0, i + 1] = y[0, i] + _h * y[1, i]
                y[1, i + 1] = y[1, i] + _h * _dvdt(t[i], [y[0, i + 1], y[1, i]])
            else:
                raise ValueError('Valid options for method are 1, 2, 3.')
        return (t, y)

    def _dvdt(t, y):
        """
        Returns dvdt at `t` given state `y`.
        """
        d = 0.5
        g = 10
        return -g / d * np.sin(y[0])

    def plot(t, y, labels):
        """
        Plot data given in t, y, v with labels [title, ylabel@left, ylabel@right]
        """
        (fig, ax1) = plt.subplots(1, 1, figsize=(10, 4))
        ax1.set_title(_labels[0])
        ax1.plot(t, y[0, :], 'b', label=' ')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(u'— ' + _labels[1], color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(t, y[1, :], 'r-.', label=' ')
        ax2.set_ylabel(u'— ‧ ' + _labels[2], color='r')
        ax2.tick_params('y', colors='r')
        plt.tight_layout()
        plt.show()
    return euler_method, plot


@app.cell
def _(euler_method, np, plot):
    (T, _y0, _h) = (10, [45 * np.pi / 180, 0], 0.01)
    (t, theta) = euler_method(T, _y0, _h, method=2)
    _labels = ['Trajectory of simple pendulum under gravity', 'Angular position ($^o$)', 'Angular velocity ($^o/s$)']
    plot(t, np.rad2deg(theta), _labels)
    return (t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Python code to automate the calculation of the Euler–Lagrange equation

        The three derivatives in the Euler–Lagrange equations are first-order derivatives and behind the scenes we are using latex to write the equations. Both tasks are boring and error prone.  
        Let's write a function using the Sympy library to automate the calculation of the derivative terms in the Euler–Lagrange equations and display them nicely.
        """
    )
    return


@app.cell
def _(Eq, Math, display, latex, mlatex, nonlinsolve, simplify, t):
    # helping function
    def printeq(lhs, rhs=None):
        """Rich display of Sympy expression as lhs = rhs."""
        if rhs is None:
            display(Math(r'{}'.format(lhs)))
        else:
            display(Math(r'{} = '.format(lhs) + mlatex(simplify(rhs, ratio=1.7))))


    def lagrange_terms(L, q, show=True):
        """Calculate terms of Euler-Lagrange equations given the Lagrangian and q's.
        """
        if not isinstance(q, list):
            q = [q]
        Lterms = []
        if show:
            s = '' if len(q) == 1 else 's'
            printeq(r"\text{Terms of the Euler-Lagrange equation%s:}"%(s))
        for qi in q:
            dLdqi = simplify(L.diff(qi))
            Lterms.append(dLdqi)
            dLdqdi = simplify(L.diff(qi.diff(t)))
            Lterms.append(dLdqdi)
            dtdLdqdi = simplify(dLdqdi.diff(t))
            Lterms.append(dtdLdqdi)
            if show:
                printeq(r'\text{For generalized coordinate}\;%s:'%latex(qi.func))
                printeq(r'\quad\dfrac{\partial\mathcal{L}}{\partial %s}'%latex(qi.func), dLdqi)
                printeq(r'\quad\dfrac{\partial\mathcal{L}}{\partial\dot{%s}}'%latex(qi.func), dLdqdi)
                printeq(r'\quad\dfrac{\mathrm d}{\mathrm{dt}}\left({\dfrac{'+
                        r'\partial\mathcal{L}}{\partial\dot{%s}}}\right)'%latex(qi.func), dtdLdqdi)
        return Lterms


    def lagrange_eq(Lterms, Qnc=None):
        """Display Euler-Lagrange equation given the Lterms."""
        s = '' if len(Lterms) == 3 else 's'
        if Qnc is None:
            Qnc = int(len(Lterms)/3) * [0]
        printeq(r"\text{Euler-Lagrange equation%s (EOM):}"%(s))
        for i in range(int(len(Lterms)/3)):
            #display(Eq(simplify(Lterms[3*i+2]-Lterms[3*i]), Qnc[i], evaluate=False))
            printeq(r'\quad ' + mlatex(simplify(Lterms[3*i+2]-Lterms[3*i])), Qnc[i])


    def lagrange_eq_solve(Lterms, q, Qnc=None):
        """Display Euler-Lagrange equation given the Lterms."""
        if not isinstance(q, list):
            q = [q]
        if Qnc is None:
            Qnc = int(len(Lterms)/3) * [0]
        system = [simplify(Lterms[3*i+2]-Lterms[3*i]-Qnc[i]) for i in range(len(q))]
        qdds = [qi.diff(t, 2) for qi in q]
        sol = nonlinsolve(system, qdds)
        s = '' if len(Lterms) == 3 else 's'
        printeq(r"\text{Euler-Lagrange equation%s (EOM):}"%(s))
        if len(sol.args):
            for i in range(int(len(Lterms)/3)):
                display(Eq(qdds[i], simplify(sol.args[0][i]), evaluate=False))
        else:
            display(sol)

        return sol
    return lagrange_eq, lagrange_eq_solve, lagrange_terms, printeq


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's recalculate the EOM of the simple pendulum using Sympy and the code for automation.
        """
    )
    return


@app.cell
def _(dynamicsymbols, sym):
    t_1 = sym.Symbol('t')
    (m, d, g) = sym.symbols('m, d, g', positive=True)
    θ = dynamicsymbols('theta')
    return d, g, m, t_1, θ


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Position and velocity of the simple pendulum under the influence of gravity:
        """
    )
    return


@app.cell
def _(cos, d, printeq, sin, t_1, θ):
    (x, y) = (d * sin(θ), -d * cos(θ))
    (xd, yd) = (x.diff(t_1), y.diff(t_1))
    printeq('x', x)
    printeq('y', y)
    printeq('\\dot{x}', xd)
    printeq('\\dot{y}', yd)
    return xd, y, yd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Kinetic and potential energies of the simple pendulum under the influence of gravity and the corresponding Lagrangian function:
        """
    )
    return


@app.cell
def _(g, m, printeq, xd, y, yd):
    T_1 = m * (xd ** 2 + yd ** 2) / 2
    V = m * g * y
    printeq('T', T_1)
    printeq('V', V)
    L = T_1 - V
    printeq('\\mathcal{L}', L)
    return (L,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the automated part for the derivatives:
        """
    )
    return


@app.cell
def _(L, lagrange_terms, θ):
    Lterms = lagrange_terms(L, θ)
    return (Lterms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, the EOM is:
        """
    )
    return


@app.cell
def _(Lterms, lagrange_eq):
    lagrange_eq(Lterms)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And rearranging:
        """
    )
    return


@app.cell
def _(Lterms, lagrange_eq_solve, θ):
    sol = lagrange_eq_solve(Lterms, q=θ, Qnc=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Same result as before.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Double pendulum under the influence of gravity

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/doublependulum_lagrange.png?raw=1" width="200" alt="double pendulum" style="float:right;margin: 10px 50px 10px 50px;"/></figure>

        Consider a double pendulum (one pendulum attached to another) with massless rods of length$d_1$and$d_2$and masses$m_1$and$m_2$at the extremities of each rod swinging in a plane forming the angles$\theta_1$and$\theta_2$with vertical.  
        The system has two particles with two degrees of freedom; two adequate generalized coordinates to describe the system's configuration are the angles in relation to the vertical ($\theta_1, \theta_2$). Let's adopt the origin of the reference frame at the point of the upper pendulum suspension.

        Let's use Sympy to solve this problem.
        """
    )
    return


@app.cell
def _(Symbol, dynamicsymbols, symbols):
    t_2 = Symbol('t')
    (d1, d2, m1, m2, g_1) = symbols('d1, d2, m1, m2, g', positive=True)
    (θ1, θ2) = dynamicsymbols('theta1, theta2')
    return d1, d2, g_1, m1, m2, t_2, θ1, θ2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The positions and velocities of masses$m_1$and$m_2$are:
        """
    )
    return


@app.cell
def _(cos, d1, d2, printeq, sin, t_2, θ1, θ2):
    _x1 = d1 * sin(θ1)
    y1 = -d1 * cos(θ1)
    _x2 = d1 * sin(θ1) + d2 * sin(θ2)
    y2 = -d1 * cos(θ1) - d2 * cos(θ2)
    (x1d, y1d) = (_x1.diff(t_2), y1.diff(t_2))
    (x2d, y2d) = (_x2.diff(t_2), y2.diff(t_2))
    printeq('x_1', _x1)
    printeq('y_1', y1)
    printeq('x_2', _x2)
    printeq('y_2', y2)
    printeq('\\dot{x}_1', x1d)
    printeq('\\dot{y}_1', y1d)
    printeq('\\dot{x}_2', x2d)
    printeq('\\dot{y}_2', y2d)
    return x1d, x2d, y1, y1d, y2, y2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The kinetic and potential energies of the system are:
        """
    )
    return


@app.cell
def _(g_1, m1, m2, printeq, x1d, x2d, y1, y1d, y2, y2d):
    T_2 = m1 * (x1d ** 2 + y1d ** 2) / 2 + m2 * (x2d ** 2 + y2d ** 2) / 2
    V_1 = m1 * g_1 * y1 + m2 * g_1 * y2
    printeq('T', T_2)
    printeq('V', V_1)
    return T_2, V_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The Lagrangian function is:
        """
    )
    return


@app.cell
def _(T_2, V_1, printeq):
    L_1 = T_2 - V_1
    printeq('\\mathcal{L}', L_1)
    return (L_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the derivatives are:
        """
    )
    return


@app.cell
def _(L_1, lagrange_terms, θ1, θ2):
    Lterms_1 = lagrange_terms(L_1, [θ1, θ2])
    return (Lterms_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, the EOM are:
        """
    )
    return


@app.cell
def _(Lterms_1, lagrange_eq):
    lagrange_eq(Lterms_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The EOM's are a system with two coupled equations,$\theta_1$and$\theta_2$appear on both equations.  

        The motion of a double pendulum is very interesting; most of times it presents a chaotic behavior.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Numerical solution of the equation of motion for the double pendulum

        The analytical solution in infeasible to deduce. For the numerical solution, first we have to rearrange the equations to find separate expressions for$\theta_1$and$\theta_2$(solve the system of equations algebraically).  
        Using Sympy, here are the two expressions:
        """
    )
    return


@app.cell
def _(Lterms_1, lagrange_eq_solve, θ1, θ2):
    sol_1 = lagrange_eq_solve(Lterms_1, q=[θ1, θ2], Qnc=None)
    return (sol_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In order to solve numerically the ODEs for the double pendulum we have to transform each equation above into two first ODEs. But we should avoid using Euler's method because of the non-negligible error in the numerical integration in this case; more accurate methods such as [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) should be employed. See such solution in [https://www.myphysicslab.com/pendulum/double-pendulum-en.html](https://www.myphysicslab.com/pendulum/double-pendulum-en.html).

        We can use Sympy to transform the symbolic equations into Numpy functions that can be used for the numerical solution. Here is the code for that:
        """
    )
    return


@app.cell
def _(d1, d2, g_1, m1, m2, sol_1, sym, t_2, θ1, θ2):
    θ1dd_fun = sym.lambdify((g_1, m1, d1, θ1, θ1.diff(t_2), m2, d2, θ2, θ2.diff(t_2)), sol_1.args[0][0], 'numpy')
    θ2dd_fun = sym.lambdify((g_1, m1, d1, θ1, θ1.diff(t_2), m2, d2, θ2, θ2.diff(t_2)), sol_1.args[0][1], 'numpy')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The reader is invited to write the code for the numerical simulation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Double compound pendulum under the influence of gravity

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/pendula_lagrange.png?raw=1" width="200" alt="double pendulum" style="float:right;margin: 10px 50px 10px 50px;"/></figure>

        Consider the double compound pendulum (or physical pendulum) shown on the the right with length$d$and mass$m$of each rod swinging in a plane forming the angles$\theta_1$and$\theta_2$with vertical and$g=10 m/s^2$.  
        The system has two degrees of freedom and we need two generalized coordinates ($\theta_1, \theta_2$) to describe the system's configuration.  

        Let's use the Lagrangian mechanics to derive the equations of motion for each pendulum.  

        To calculate the potential and kinetic energy of the system, we will need to calculate the position and velocity of each pendulum. Now each pendulum is a rod with distributed mass and we will have to calculate the moment of rotational inertia of the rod. In this case, the kinetic energy of each pendulum will be given as the kinetic energy due to rotation of the pendulum plus the kinetic energy due to the speed of the center of mass of the pendulum, such that the total kinetic energy of the system is:$\begin{array}{rcl}
        T = \overbrace{\underbrace{\,\frac{1}{2}I_{1,cm}\dot\theta_1^2\,}_{\text{rotation}} + \underbrace{\frac{1}{2}m(\dot x_{1,cm}^2 + \dot y_{1,cm}^2)}_{\text{translation}}}^{\text{pendulum 1}} + \overbrace{\underbrace{\,\frac{1}{2}I_{2,cm}\dot\theta_2^2\,}_{\text{rotation}} + \underbrace{\frac{1}{2}m(\dot x_{2,cm}^2 + \dot y_{2,cm}^2)}_{\text{translation}}}^{\text{pendulum 2}}
        \end{array}$And the potential energy of the system is:$\begin{array}{rcl}
        V = mg\big(y_{1,cm} + y_{2,cm}\big)
        \end{array}$Let's use Sympy once again.

        The position and velocity of the center of mass of the rods$1$and$2$are:
        """
    )
    return


@app.cell
def _(cos, dynamicsymbols, printeq, sin, symbols, t_2):
    (d_1, m_1, g_2) = symbols('d, m, g', positive=True)
    (θ1_1, θ2_1) = dynamicsymbols('theta1, theta2')
    I = m_1 * d_1 * d_1 / 12
    _x1 = d_1 * sin(θ1_1) / 2
    y1_1 = -d_1 * cos(θ1_1) / 2
    _x2 = d_1 * sin(θ1_1) + d_1 * sin(θ2_1) / 2
    y2_1 = -d_1 * cos(θ1_1) - d_1 * cos(θ2_1) / 2
    (x1d_1, y1d_1) = (_x1.diff(t_2), y1_1.diff(t_2))
    (x2d_1, y2d_1) = (_x2.diff(t_2), y2_1.diff(t_2))
    printeq('x_1', _x1)
    printeq('y_1', y1_1)
    printeq('x_2', _x2)
    printeq('y_2', y2_1)
    printeq('\\dot{x}_1', x1d_1)
    printeq('\\dot{y}_1', y1d_1)
    printeq('\\dot{x}_2', x2d_1)
    printeq('\\dot{y}_2', y2d_1)
    return I, g_2, m_1, x1d_1, x2d_1, y1_1, y1d_1, y2_1, y2d_1, θ1_1, θ2_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The kinetic and potential energies of the system are:
        """
    )
    return


@app.cell
def _(
    I,
    g_2,
    m_1,
    printeq,
    t_2,
    x1d_1,
    x2d_1,
    y1_1,
    y1d_1,
    y2_1,
    y2d_1,
    θ1_1,
    θ2_1,
):
    T_3 = I / 2 * θ1_1.diff(t_2) ** 2 + m_1 / 2 * (x1d_1 ** 2 + y1d_1 ** 2) + I / 2 * θ2_1.diff(t_2) ** 2 + m_1 / 2 * (x2d_1 ** 2 + y2d_1 ** 2)
    V_2 = m_1 * g_2 * y1_1 + m_1 * g_2 * y2_1
    printeq('T', T_3)
    printeq('V', V_2)
    return T_3, V_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The Lagrangian function is:
        """
    )
    return


@app.cell
def _(T_3, V_2, printeq):
    L_2 = T_3 - V_2
    printeq('\\mathcal{L}', L_2)
    return (L_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the derivatives are:
        """
    )
    return


@app.cell
def _(L_2, lagrange_terms, θ1_1, θ2_1):
    Lterms_2 = lagrange_terms(L_2, [θ1_1, θ2_1])
    return (Lterms_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, the EOM are:
        """
    )
    return


@app.cell
def _(Lterms_2, lagrange_eq):
    lagrange_eq(Lterms_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And rearranging:
        """
    )
    return


@app.cell
def _(Lterms_2, lagrange_eq_solve, θ1_1, θ2_1):
    sol_2 = lagrange_eq_solve(Lterms_2, q=[θ1_1, θ2_1], Qnc=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Double compound pendulum in joint space

        Let's recalculate the former example but employing generalized coordinates in the joint space:$\alpha_1=\theta_1$and$\alpha_2=\theta_2-\theta_1$.
        """
    )
    return


@app.cell
def _(cos, dynamicsymbols, printeq, sin, symbols, t_2):
    (d_2, m_2, g_3) = symbols('d, m, g', positive=True)
    (α1, α2) = dynamicsymbols('alpha1, alpha2')
    I_1 = m_2 * d_2 * d_2 / 12
    _x1 = d_2 * sin(α1) / 2
    y1_2 = -d_2 * cos(α1) / 2
    _x2 = d_2 * sin(α1) + d_2 * sin(α1 + α2) / 2
    y2_2 = -d_2 * cos(α1) - d_2 * cos(α1 + α2) / 2
    (x1d_2, y1d_2) = (_x1.diff(t_2), y1_2.diff(t_2))
    (x2d_2, y2d_2) = (_x2.diff(t_2), y2_2.diff(t_2))
    printeq('x_1', _x1)
    printeq('y_1', y1_2)
    printeq('x_2', _x2)
    printeq('y_2', y2_2)
    printeq('\\dot{x}_1', x1d_2)
    printeq('\\dot{y}_1', y1d_2)
    printeq('\\dot{x}_2', x2d_2)
    printeq('\\dot{y}_2', y2d_2)
    return I_1, g_3, m_2, x1d_2, x2d_2, y1_2, y1d_2, y2_2, y2d_2, α1, α2


@app.cell
def _(
    I_1,
    g_3,
    m_2,
    printeq,
    t_2,
    x1d_2,
    x2d_2,
    y1_2,
    y1d_2,
    y2_2,
    y2d_2,
    α1,
    α2,
):
    T_4 = I_1 / 2 * α1.diff(t_2) ** 2 + m_2 / 2 * (x1d_2 ** 2 + y1d_2 ** 2) + I_1 / 2 * (α1.diff(t_2) + α2.diff(t_2)) ** 2 + m_2 / 2 * (x2d_2 ** 2 + y2d_2 ** 2)
    V_3 = m_2 * g_3 * y1_2 + m_2 * g_3 * y2_2
    L_3 = T_4 - V_3
    printeq('T', T_4)
    printeq('V', V_3)
    printeq('\\mathcal{L}', L_3)
    return (L_3,)


@app.cell
def _(L_3, lagrange_terms, α1, α2):
    Lterms_3 = lagrange_terms(L_3, [α1, α2])
    return (Lterms_3,)


@app.cell
def _(Lterms_3, lagrange_eq):
    lagrange_eq(Lterms_3)
    return


@app.cell
def _(Lterms_3, lagrange_eq_solve, α1, α2):
    sol_3 = lagrange_eq_solve(Lterms_3, q=[α1, α2], Qnc=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Forces on the pendulum**

        We can see that besides the terms proportional to gravity$g$, there are three types of forces in the equations, two of these forces we already saw in the solution of the double pendulum employing generalized coordinates in the segment space: forces proportional to angular velocity squared$\dot{\theta}_i^2$(now proportional to$\dot{\alpha}_i^2$) and forces proportional to the angular acceleration$\ddot{\theta}_i$(now proportional to$\ddot{\alpha}_i$). These are the centripetal forces and tangential forces.  
        A new type of force appeared explicitly in the equations when we employed generalized coordinates in the joint space: forces proportional to the product of the two angular velocities in joint space$\dot{\alpha}_1\dot{\alpha}_2$. These are the force of Coriolis.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Mass attached to a spring on a horizontal plane

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/ruina_prob13_1_7.png?raw=1" width="300" alt="mass-spring on a table" style="float:right;margin: 0px 0px 0px 5px;"/></figure>

        Let's solve the exercise 13.1.7 of Ruina and Pratap (2019):  
        "Two ice skaters whirl around one another. They are connected by a linear elastic cord whose center is stationary in space. We wish to consider the motion of one of the skaters by modeling her as a mass m held by a cord that exerts k Newtons for each meter it is extended from the central position.  
        a) Draw a free-body diagram showing the forces that act on the mass is at an arbitrary position.  
        b) Write the differential equations that describe the motion."

        Let's solve item b using Lagrangian mechanics.

        To calculate the potential and kinetic energy of the system, we will need to calculate the position and velocity of the mass. It's convenient to use as generalized coordinates, the radial position$r$and the angle$\theta$.

        Using Sympy, declaring our parameters and coordinates:
        """
    )
    return


@app.cell
def _(Symbol, dynamicsymbols, symbols):
    t_3 = Symbol('t')
    (m_3, k) = symbols('m, k', positive=True)
    (r, θ_1) = dynamicsymbols('r, theta')
    return k, m_3, r, t_3, θ_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The position and velocity of the skater are:
        """
    )
    return


@app.cell
def _(cos, printeq, r, sin, t_3, θ_1):
    (x_1, y_1) = (r * cos(θ_1), r * sin(θ_1))
    (xd_1, yd_1) = (x_1.diff(t_3), y_1.diff(t_3))
    printeq('x', x_1)
    printeq('y', y_1)
    printeq('\\dot{x}', xd_1)
    printeq('\\dot{y}', yd_1)
    return xd_1, yd_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        So, the kinetic and potential energies of the skater are:
        """
    )
    return


@app.cell
def _(Math, display, k, m_3, mlatex, printeq, r, xd_1, yd_1):
    T_5 = m_3 * (xd_1 ** 2 + yd_1 ** 2) / 2
    V_4 = k * r ** 2 / 2
    display(Math('T=' + mlatex(T_5)))
    display(Math('V=' + mlatex(V_4)))
    printeq('T', T_5)
    printeq('V', V_4)
    return T_5, V_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Where we considered the equilibrium length of the spring as zero.

        The Lagrangian function is:
        """
    )
    return


@app.cell
def _(T_5, V_4, printeq):
    L_4 = T_5 - V_4
    printeq('\\mathcal{L}', L_4)
    return (L_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the derivatives are:
        """
    )
    return


@app.cell
def _(L_4, lagrange_terms, r, θ_1):
    Lterms_4 = lagrange_terms(L_4, [r, θ_1])
    return (Lterms_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, the EOM are:
        """
    )
    return


@app.cell
def _(Lterms_4, lagrange_eq):
    lagrange_eq(Lterms_4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In Ruina and Pratap's book they give as solution the equation:$2r\dot{r}\dot{\theta} + r^3\ddot{\theta}=0$, but  using dimensional analysis we can check the book's solution is not correct.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Generalized forces

        How non-conservative forces are treated in the Lagrangian Mechanics is different than in Newtonian mechanics.  
        Newtonian mechanics consider the forces (and moment of forces) acting on each body (via FBD) and write down the equations of motion for each body/coordinate.  
        In Lagrangian Mechanics, we consider the forces (and moment of forces) acting on each generalized coordinate. For such, the effects of the non-conservative forces have to be calculated in the direction of each generalized coordinate, these will be the generalized forces.  

        A robust approach to determine the generalized forces on each generalized coordinate is to compute the work done by the forces to produce a small variation of the system on the direction of the generalized coordinate.  

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/pendulumforce.png?raw=1" width="260" alt="pendulum" style="float:right;margin: 5px 10px 5px 5px;"/></figure>

        For example, consider a pendulum with a massless rod of length$d$and a mass$m$at the extremity swinging in a plane forming the angle$\theta$with the vertical.   
        An external force acts on the tip of the pendulum at the horizontal direction.  
        The pendulum cord is inextensible and the tip of the pendulum can only move along the arc of a circumference with radius$d$.  

        The work done by this force to produce a small variation$\delta \theta$is:
        <p>
        <span class="notranslate">$\begin{array}{l}
        \delta W_{NC} = \vec{F} \cdot \delta \vec{r} \\
        \delta W_{NC} = F d \cos(\theta) \delta \theta
        \end{array}$</span>
       
        We now reexpress the work as the product of the corresponding generalized force$Q_{NC}$and the generalized coordinate:
        <p>
        <span class="notranslate">$\delta W_{NC} = Q_{NC} \delta \theta$</span>

        And comparing the last two equations, the generalized force (in fact, a moment of force) is:
        <p>
        <span class="notranslate">$Q_{NC} = F d \cos(\theta)$</span>

        Note that the work done by definition was expressed in Cartesian coordinates as the scalar product between vectors$\vec F$and$\delta \vec{r}$and after the scalar product was evaluated we end up with the work done expressed in terms of the generalized coordinate. This is somewhat similar to the calculation of kinetic and potential energy, these quantities are typically expressed first in terms of Cartesian coordinates, which in turn are expressed in terms of the generalized coordinates, so we end up with only generalized coordinates.  
        Also note, we employ the term generalized force to refer to a non-conservative force or moment of force expressed in the generalized coordinate.
    
        If the force had components on both directions, we would calculate the work computing the scalar product between the variation in displacement and the force, as usual. For example, consider a force$\vec{F}=2\hat{i}+7\hat{j}$, the work done is:
        <p>
        <span class="notranslate">$\begin{array}{l}
        \delta W_{NC} = \vec{F} \cdot \delta \vec{r} \\
        \delta W_{NC} = [2\hat{i}+7\hat{j}] \cdot [d\cos(\theta) \delta \theta \hat{i} + d\sin(\theta) \delta \theta \hat{j}] \\
        \delta W_{NC} = d[2\cos(\theta) + 7\sin(\theta)] \delta \theta  
        \end{array}$</span>

        Finally, the generalized force (a moment of force) is:
        <p>
        <span class="notranslate">$Q_{NC} = d[2\cos(\theta) + 7\sin(\theta)]$</span>    

        For a system with$N$generalized coordinates and$n$non-conservative forces, to determine the generalized force at each generalized coordinate, we would compute the work as the sum of the works done by each force at each small variation:
        <p>
        <span class="notranslate">$\delta W_{NC} = \sum\limits_{j=1}^n F_{j} \cdot \delta x_j(\delta q_1, \dotsc, \delta q_N ) = \sum\limits_{i=1}^N  Q_{i} \delta q_i$</span>

        For simpler problems, in which we can separately analyze each non-conservative force acting on each generalized coordinate, the work done by each force on a given generalized coordinate can be calculated by making all other coordinates immovable ('frozen') and then sum the generalized forces.  

        The next examples will help to understand how to calculate the generalized force.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Simple pendulum on moving cart

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/masspend_lagrange.png?raw=1" width="250" alt="cart pendulum" style="float:right;margin: 10px 50px 10px 50px;"/></figure>

        Consider a simple pendulum with massless rod of length$d$and mass$m$at the extremity of the rod forming an angle$\theta$with the vertical direction under the action of gravity. The pendulum swings freely from a cart with mass$M$that moves at the horizontal direction pushed by a force$F_x$.  

        Let's use the Lagrangian mechanics to derive the EOM for the system.

        We will model the cart as a particle moving along the axis$x$, i.e.,$y=0$. The system has two degrees of freedom and because of the constraints introduced by the constant length of the rod and the motion the cart can perform, good generalized coordinates to describe the configuration of the system are$x$and$\theta$.  

        **Determination of the generalized force**  
        The force$F$acts along the same direction of the generalized coordinate$x$, this means$F$contributes entirely to the work done at the direction$x$. At this generalized coordinate, the generalized force due to$F$is$F$.  
        At the generalized coordinate$θ$, if we 'freeze' the generalized coordinate$x$and let$F$act on the system, there is no movement at the generalized coordinate$θ$, so no work is done. At this generalized coordinate, the generalized force due to$F$is$0$.

        Let's now use Sympy to determine the EOM.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The positions of the cart (c) and of the pendulum tip (p) are:  
        """
    )
    return


@app.cell
def _(Symbol, dynamicsymbols, symbols):
    t_4 = Symbol('t')
    (M, m_4, d_3) = symbols('M, m, d', positive=True)
    (x_2, y_2, θ_2, Fx) = dynamicsymbols('x, y, theta, F_x')
    return Fx, M, d_3, m_4, t_4, x_2, y_2, θ_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The positions of the cart (c) and of the pendulum tip (p) are:
        """
    )
    return


@app.cell
def _(cos, d_3, printeq, sin, t_4, x_2, y_2, θ_2):
    (xc, yc) = (x_2, y_2 * 0)
    (xcd, ycd) = (xc.diff(t_4), yc.diff(t_4))
    (xp, yp) = (x_2 + d_3 * sin(θ_2), -d_3 * cos(θ_2))
    (xpd, ypd) = (xp.diff(t_4), yp.diff(t_4))
    printeq('x_c', xc)
    printeq('y_c', yc)
    printeq('x_p', xp)
    printeq('y_p', yp)
    return xcd, xpd, yc, ycd, yp, ypd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The velocities of the cart and of the pendulum are:
        """
    )
    return


@app.cell
def _(printeq, xcd, xpd, ycd, ypd):
    printeq(r'\dot{x}_c', xcd)
    printeq(r'\dot{y}_c', ycd)

    printeq(r'\dot{x}_p', xpd)
    printeq(r'\dot{y}_p', ypd)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The total kinetic and total potential energies and the Lagrangian of the system are:
        """
    )
    return


@app.cell
def _(M, g_3, m_4, printeq, xcd, xpd, yc, ycd, yp, ypd):
    T_6 = M * (xcd ** 2 + ycd ** 2) / 2 + m_4 * (xpd ** 2 + ypd ** 2) / 2
    V_5 = M * g_3 * yc + m_4 * g_3 * yp
    printeq('T', T_6)
    printeq('V', V_5)
    L_5 = T_6 - V_5
    printeq('\\mathcal{L}', L_5)
    return (L_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the derivatives are:
        """
    )
    return


@app.cell
def _(L_5, lagrange_terms, x_2, θ_2):
    Lterms_5 = lagrange_terms(L_5, [x_2, θ_2])
    return (Lterms_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, the EOM are:
        """
    )
    return


@app.cell
def _(Fx, Lterms_5, lagrange_eq):
    lagrange_eq(Lterms_5, [Fx, 0])
    return


@app.cell
def _(Fx, Lterms_5, lagrange_eq_solve, x_2, θ_2):
    sol_4 = lagrange_eq_solve(Lterms_5, q=[x_2, θ_2], Qnc=[Fx, 0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that although the force$F_x$acts solely on the cart, the acceleration of the pendulum$\ddot{\theta}$is also dependent on$F$, as expected.  

        [Clik here for solutions to this problem using the Newtonian and Lagrangian approaches and how this system of two coupled second order differential equations can be rearranged for its numerical solution](http://www.emomi.com/download/neumann/pendulum_cart.html).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Two masses and two springs under the influence of gravity

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/springs_masses_g.png?raw=1" width="200" alt="double pendulum" style="float:right;margin: 10px 50px 10px 50px;"/></figure>

        Consider a system composed by two masses$m_1,\, m_2$and two ideal springs (massless, lengths$\ell_1,\, \ell_2$, and spring constants$k_1,\,k_2$) attached in series under gravity and a force$F$acting directly on$m_2$.  

        We can model this system as composed by two particles with two degrees of freedom and we need two generalized coordinates to describe the system's configuration; two obvious options are:  

         -${y_1, y_2}$, positions of masses$m_1,\, m_2$w.r.t. ceiling (origin).  
         -${z_1, z_2}$, position of mass$m_1$w.r.t. ceiling and position of mass$m_2$w.r.t. mass$m_1$.

        The set${y_1, y_2}$is at an inertial reference frame, while the second set it's not.  
        Let's find the EOM's using both sets of generalized coordinates and compare them.

        **Generalized forces**  
        Using${y_1, y_2}$, force$F$acts on mass$m_2$at the same direction of the generalized coordinate$y_2$. At this coordinate, the generalized force of$F$is$F$. At the generalized coordinate$y_1$, if we 'freeze' the generalized coordinate$y_2$and let$F$act on the system, there is no movement at the generalized coordinate$y_1$and no work is done. At this generalized coordinate, the generalized force due to$F$is$0$.  

        Using${z_1, z_2}$, force$F$acts on mass$m_2$at the same direction of the generalized coordinate$z_2$. At this coordinate, the generalized force of$F$is$F$. At the generalized coordinate$z_1$, if we 'freeze' the generalized coordinate$z_2$and let$F$act on the system, mass$m_1$suffers the action of force$F$at the generalized coordinate$y_1$. At this generalized coordinate, the generalized force due to$F$is$F$.

        Sympy is our friend once again:
        """
    )
    return


@app.cell
def _(Symbol, dynamicsymbols, symbols):
    t_5 = Symbol('t')
    (m1_1, m2_1, l01, l02, g_4, k1, k2) = symbols('m1, m2, ell01, ell02, g, k1, k2', positive=True)
    (y1_3, y2_3, F) = dynamicsymbols('y1, y2, F')
    return F, g_4, k1, k2, l01, l02, m1_1, m2_1, t_5, y1_3, y2_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The total kinetic and total potential energies of the system are:
        """
    )
    return


@app.cell
def _(g_4, k1, k2, l01, l02, m1_1, m2_1, printeq, t_5, y1_3, y2_3):
    (y1d_3, y2d_3) = (y1_3.diff(t_5), y2_3.diff(t_5))
    T_7 = m1_1 * y1d_3 ** 2 / 2 + m2_1 * y2d_3 ** 2 / 2
    V_6 = k1 * (y1_3 - l01) ** 2 / 2 + k2 * (y2_3 - y1_3 - l02) ** 2 / 2 - m1_1 * g_4 * y1_3 - m2_1 * g_4 * y2_3
    printeq('T', T_7)
    printeq('V', V_6)
    return T_7, V_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For sake of clarity, let's consider the resting lengths of the springs to be zero:
        """
    )
    return


@app.cell
def _(V_6, l01, l02, printeq):
    V_7 = V_6.subs([(l01, 0), (l02, 0)])
    printeq('V', V_7)
    return (V_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The Lagrangian function is:
        """
    )
    return


@app.cell
def _(T_7, V_7, printeq):
    L_6 = T_7 - V_7
    printeq('\\mathcal{L}', L_6)
    return (L_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the derivatives are:
        """
    )
    return


@app.cell
def _(L_6, lagrange_terms, y1_3, y2_3):
    Lterms_6 = lagrange_terms(L_6, [y1_3, y2_3])
    return (Lterms_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, the EOM are:
        """
    )
    return


@app.cell
def _(F, Lterms_6, lagrange_eq):
    lagrange_eq(Lterms_6, [0, F])
    return


@app.cell
def _(F, Lterms_6, lagrange_eq_solve, y1_3, y2_3):
    lagrange_eq_solve(Lterms_6, [y1_3, y2_3], [0, F])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Same problem, but with the other set of coordinates**

        Using${z_1, z_2}$as the position of mass$m_1$w.r.t. the ceiling and the position of mass$m_2$w.r.t. the mass$m_1$, the solution is:
        """
    )
    return


@app.cell
def _(dynamicsymbols, g_4, k1, k2, l01, l02, m1_1, m2_1, printeq, t_5):
    (z1, z2) = dynamicsymbols('z1, z2')
    (z1d, z2d) = (z1.diff(t_5), z2.diff(t_5))
    T_8 = m1_1 * z1d ** 2 / 2 + m2_1 * (z1d + z2d) ** 2 / 2
    V_8 = k1 * (z1 - l01) ** 2 / 2 + k2 * (z2 - l02) ** 2 / 2 - m1_1 * g_4 * z1 - m2_1 * g_4 * (z1 + z2)
    printeq('T', T_8)
    printeq('V', V_8)
    return T_8, V_8, z1, z2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For sake of clarity, let's consider the resting lengths of the springs to be zero:
        """
    )
    return


@app.cell
def _(V_8, l01, l02, printeq):
    V_9 = V_8.subs([(l01, 0), (l02, 0)])
    printeq('V', V_9)
    return (V_9,)


@app.cell
def _(T_8, V_9, printeq):
    L_7 = T_8 - V_9
    printeq('\\mathcal{L}', L_7)
    return (L_7,)


@app.cell
def _(F, L_7, lagrange_eq, lagrange_terms, z1, z2):
    Lterms_7 = lagrange_terms(L_7, [z1, z2])
    lagrange_eq(Lterms_7, [F, F])
    return (Lterms_7,)


@app.cell
def _(F, Lterms_7, lagrange_eq_solve, z1, z2):
    lagrange_eq_solve(Lterms_7, [z1, z2], [F, F])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The solutions using the two sets of coordinates seem different; the reader is invited to verify that in fact they are the same (remember that$y_1 = z_1,\, y_2 = z_1+z_2,\, \ddot{y}_2 = \ddot{z}_1+\ddot{z}_2$).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Mass-spring-damper system with gravity

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/mass_spring_damper_gravity.png?raw=1" width="220" alt="mass-spring-damper system" style="float:right;margin: 10px 50px 10px 50px;"/></figure>

        Consider a mass-spring-damper system under the action of the gravitational force and an external force acting on the mass.  
        The massless spring has a stiffness coefficient$k$and length at rest$\ell_0$.  
        The massless damper has a damping coefficient$b$.  
        The gravitational force acts downwards and it is negative (see figure).  
        The system has one degree of freedom and we need only one generalized coordinate ($y$) to describe the system's configuration.  

        There are two non-conservative forces acting at the direction of the generalized coordinate: the external force F and the force of the damper. By calculating the work done by each of these forces, the total generalized force is:$Q_{NC} = F_0 \cos(\omega t) - b\dot y$.

        Let's use the Lagrangian mechanics to derive the equations of motion for the system.

        The kinetic energy of the system is:$T = \frac{1}{2} m \dot y^2$The potential energy of the system is:$V = \frac{1}{2} k (y-\ell_0)^2 + m g y$The Lagrangian function is:$\mathcal{L} = \frac{1}{2} m \dot y^2 - \frac{1}{2} k (y-\ell_0)^2 - m g y$The derivatives of the Lagrangian w.r.t.$y$and$t$are:$\begin{array}{rcl}
        \dfrac{\partial \mathcal{L}}{\partial y} &=& -k(y-\ell_0) - mg \\
        \dfrac{\partial \mathcal{L}}{\partial \dot{y}} &=& m \dot{y} \\
        \dfrac{\mathrm d }{\mathrm d t}\left( {\dfrac{\partial \mathcal{L}}{\partial \dot{y}}} \right) &=& m\ddot{y}
        \end{array}$Substituting all these terms in the Euler-Lagrange equation, results in:$m\ddot{y} + b\dot{y} + k(y-\ell_0) + mg = F_0 \cos(\omega t)$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Numerical solution of the equation of motion for mass-spring-damper system

        Let's solve numerically the differential equation for the mass-spring-damper system with gravity using the function for the Euler's method we implemented before.  We just have to write a new function for calculating the derivative of velocity:
        """
    )
    return


@app.cell
def _(euler_method, np, plot):
    def _dvdt(t, y):
        """
        Returns dvdt at `t` given state `y`.
        """
        m = 1
        k = 100
        l0 = 1.0
        b = 1.0
        F0 = 2.0
        f = 1
        g = 10
        F = F0 * np.cos(2 * np.pi * f * t)
        return (F - b * y[1] - k * (y[0] - l0) - m * g) / m
    (T_9, _y0, _h) = (10, [1.1, 0], 0.01)
    (t_6, y_3) = euler_method(T_9, _y0, _h, method=2)
    _labels = ['Trajectory of mass-spring-damper system (Euler method)', 'Position (m)', 'Velocity (m/s)']
    plot(t_6, y_3, _labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here is the solution for this problem using the integration method explicit Runge-Kutta, a method with smaller errors in the integration and faster (for large amount of data) because the integration method in the Scipy function is implemented in Fortran:
        """
    )
    return


@app.cell
def _(np, plot):
    from scipy.integrate import solve_ivp

    def dvdt2(t, y):
        """
        Returns dvdt at `t` given state `y`.
        """
        m = 1
        k = 100
        l0 = 1.0
        b = 1.0
        F0 = 2.0
        f = 1
        g = 10
        F = F0 * np.cos(2 * np.pi * f * t)
        return (y[1], (F - b * y[1] - k * (y[0] - l0) - m * g) / m)
    T_10 = 10.0
    freq = 100
    y02 = [1.1, 0.0]
    t_7 = np.linspace(0, T_10, int(T_10 * freq), endpoint=False)
    s = solve_ivp(fun=dvdt2, t_span=(t_7[0], t_7[-1]), y0=y02, method='RK45', t_eval=t_7)
    _labels = ['Trajectory of mass-spring-damper system (Runge-Kutta method)', 'Position (m)', 'Velocity (m/s)']
    plot(s.t, s.y, _labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Forces of constraint

        The fact the Lagrangian formalism uses generalized coordinates means that in a system with constraints we typically have fewer coordinates (in turn, fewer equations of motion) and we don't need to worry about forces of constraint that we would have to consider in the Newtonian formalism.  
        However, when we do want to determine a force of constraint, using Lagrangian formalism in fact will be disadvantageous! Let's see now one way of determining a force of constraint using Lagrangian formalism. The trick is to postpone the consideration that there is a constraint in the system; this will increase the number of generalized coordinates but will allow the determination of a force of constraint.
        Let's exemplify this approach determining the tension at the rod in the simple pendulum under the influence of gravity we saw earlier.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Force of constraint in a simple pendulum under the influence of gravity

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/simplependulum_lagrange.png?raw=1" width="220" alt="simple pendulum" style="float:right;margin: 10px 50px 10px 50px;"/></figure>

        Consider a pendulum with a massless rod of length$d$and a mass$m$at the extremity swinging in a plane forming the angle$\theta$with vertical and$g=10 m/s^2$.  

        Although the pendulum moves at the plane, it only has one degree of freedom, which can be described by the angle$\theta$, the generalized coordinate. But because we want to determine the force of constraint tension at the rod, let's also consider for now the variable$r$for the 'varying' length of the rod (instead of the constant$d$).  

        In this case, the kinetic energy of the system will be:
        <p>
        <span class="notranslate">$T = \frac{1}{2}mr^2\dot\theta^2 + \frac{1}{2}m\dot r^2$</span>
    
        And for the potential energy we will also have to consider the constraining potential,$V_r(r(t))$:
        <p>
        <span class="notranslate">$V = -mgr\cos\theta + V_r(r(t))$</span>
    
        The Lagrangian function is:
        <p>
        <span class="notranslate">$\mathcal{L} = \frac{1}{2}m(\dot r^2(t) + r^2(t)\,\dot\theta^2(t)) + mgr(t)\cos\theta(t) - V_r(r(t))$</span>
    
        The derivatives w.r.t.$\theta$are:
        <p>
        <span class="notranslate">$\begin{array}{rcl}
        &\dfrac{\partial \mathcal{L}}{\partial \theta} &=& -mgr\sin\theta \\
        &\dfrac{\partial \mathcal{L}}{\partial \dot{\theta}} &=& mr^2\dot{\theta} \\
        &\dfrac{\mathrm d }{\mathrm d t}\left( {\dfrac{\partial \mathcal{L}}{\partial \dot{\theta}}} \right) &=& 2mr\dot{r}\dot{\theta} + mr^2\ddot{\theta}
        \end{array}$</span>
    
        The derivatives w.r.t.$r$are:
        <p>
        <span class="notranslate">$\begin{array}{rcl}
        &\dfrac{\partial \mathcal{L}}{\partial r} &=& mr \dot\theta^2 + mg\cos\theta - \dot{V}_r(r) \\
        &\dfrac{\partial \mathcal{L}}{\partial \dot{r}} &=& m\dot r \\
        &\dfrac{\mathrm d }{\mathrm d t}\left( {\dfrac{\partial \mathcal{L}}{\partial \dot{r}}} \right) &=& m\ddot{r}
        \end{array}$</span>
    
        The Euler-Lagrange's equations (the equations of motion) are:
        <p>
        <span class="notranslate">$\begin{array}{rcl}
        &2mr\dot{r}\dot{\theta} + mr^2\ddot{\theta} + mgr\sin\theta &=& 0 \\
        &m\ddot{r} - mr \dot\theta^2 - mg\cos\theta + \dot{V}_r(r) &=& 0 \\
        \end{array}$</span>
    
        Now, we will apply the constraint condition,$r(t)=d$. This means that$\dot{r}=\ddot{r}=0$.  

        With this constraint applied, the first Euler-Lagrange equation is the equation for the simple pendulum:
        <p>
        <span class="notranslate">$md^2\ddot{\theta} + mgd\sin\theta = 0$</span>
    
        The second equation yields:
        <p>
        <span class="notranslate">$-\dfrac{\mathrm d V_r}{\mathrm d r}\bigg{\rvert}_{r=d} = - md \dot\theta^2 - mg\cos\theta$</span>
    
        But the tension force,$F_T$, is by definition equal to the gradient of the constraining potential, so:
        <p>
        <span class="notranslate">$F_T = - md \dot\theta^2 - mg\cos\theta$</span>
    
        As expected, the tension at the rod is proportional to the centripetal and the gravitational forces.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Lagrangian formalism applied to non-mechanical systems

        ### Example: Lagrangian formalism for RLC eletrical circuits

        <figure><img src="https://upload.wikimedia.org/wikipedia/en/thumb/6/65/Mobility_analogy_resonator_vertical.svg/198px-Mobility_analogy_resonator_vertical.svg.png" width="150" alt="RLC analogy" style="float:right;margin: 10px 10px 10px 10px;"/></figure>

        It's possible to solve a RLC (Resistance-Inductance-Capacitance) electrical circuit using the Lagrangian formalism as an analogy with a mass-spring-damper system.  

        In such analogy, the electrical charge is equivalent to position, current to velocity, inductance to mass, inverse of the capacitance to spring constant, resistance to damper constant (a dissipative element), and a generator would be analog to an external force actuating on the system. See the [Wikipedia](https://en.wikipedia.org/wiki/Mechanical%E2%80%93electrical_analogies) and [this paper](https://arxiv.org/pdf/1711.10245.pdf) for more details on this analogy.

        Let's see how to deduce the equivalent of equation of motion for a RLC series circuit (the Kirchhoff’s Voltage Law (KVL) equation).

        <figure><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/RLC_series_circuit_v1.svg/173px-RLC_series_circuit_v1.svg.png" width="140" alt="RLC circuit" style="float:right;margin: 0px 10px 0px 10px;"/></figure>

        For a series RLC circuit, consider the following notation:$q$: charge$\dot{q}$: current admitted through the circuit$R$: effective resistance of the combined load, source, and components$C$: capacitance of the capacitor component$L$: inductance of the inductor component$u$: voltage source powering the circuit$P$: power dissipated by the resistance  

        So, the equivalents of kinetic and potential energies are:$T = \frac{1}{2}L\dot{q}^2$$V = \frac{1}{2C}q^2$With a dissipative element:$P = \frac{1}{2}R\dot{q}^2$And the Lagrangian function is:$\mathcal{L} = \frac{1}{2}L\dot{q}^2 - \frac{1}{2C}q^2$Calculating the derivatives and substituting them in the Euler-Lagrange equation, we will have:  
        <p>
        <span class="notranslate">$L \ddot{q} + R\dot{q} + \frac{q}{C} = u(t)$</span>

        Replacing$\dot{q}$by$i$and considering$v_c = q/C$for a capacitor, we have the familar KVL equation:
        <p>
        <span class="notranslate">$L \frac{\mathrm d i}{\mathrm d t} + v_c + Ri = u(t)$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Considerations on the Lagrangian mechanics

        The Lagrangian mechanics does not constitute a new theory in classical mechanics; the results using Lagrangian or Newtonian mechanics must be the same for any mechanical system, only the method used to obtain the results is different.

        We are accustomed to think of mechanical systems in terms of vector quantities such as force, velocity, angular momentum, torque, etc., but in the Lagrangian formalism the equations of motion are obtained entirely in terms of the kinetic and potential energies (scalar operations) in the configuration space. Another important aspect of the force vs. energy analogy is that in situations where it is not possible to make explicit all the forces acting on the body, it is still possible to obtain expressions for the kinetic and potential energies.

        In fact, the concept of force does not enter into Lagrangian mechanics. This is an important property of the method. Since energy is a scalar quantity, the Lagrangian function for a system is invariant for coordinate transformations. Therefore, it is possible to move from a certain configuration space (in which the equations of motion can be somewhat complicated) to a space that can be chosen to allow maximum simplification of the problem.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

        - [The Principle of Least Action](https://www.feynmanlectures.caltech.edu/II_19.html)  
        - Vandiver JK (MIT OpenCourseWare) [An Introduction to Lagrangian Mechanics](https://ocw.mit.edu/courses/mechanical-engineering/2-003sc-engineering-dynamics-fall-2011/lagrange-equations/MIT2_003SCF11_Lagrange.pdf)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the internet   

        - iLectureOnline: [Lectures in Lagrangian Mechanics](http://www.ilectureonline.com/lectures/subject/PHYSICS/34/245)  
        - MIT OpenCourseWare: [Introduction to Lagrange With Examples](https://youtu.be/zhk9xLjrmi4)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. Derive the Euler-Lagrange equation (the equation of motion) for a mass-spring system where the spring is attached to the ceiling and the mass in hanging in the vertical (see solution for a system with two masses at https://youtu.be/dZjjzzWykio).  
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/springgravity.png?raw=1" width="200" alt="mass-spring with gravity"/></figure>  

        2. Derive the Euler-Lagrange equation for an inverted pendulum in the vertical.  
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/invpendulum2.png?raw=1" width="200" alt="inverted pendulum"/></figure>  

        3. Derive the Euler-Lagrange equation for the following system (see solution at https://youtu.be/8FSjEsUVNx8):  
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/massessprings_lagrange.png?raw=1" width="280" alt="two masses and two springs"/></figure>  

        4. Derive the Euler-Lagrange equation for a spring pendulum, a simple pendulum where a mass$m$is attached to a massless spring with spring constant$k$and length at rest$d_0$(see solution at https://youtu.be/iULa9A00JpA).  
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/pendulumspring.png?raw=1" width="200" alt="mass-spring pendulum"/></figure>

        5. Derive the Euler-Lagrange equation for the system shown below (see solution for one mass spring system at https://youtu.be/eY0I8QK-ITE).  
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/pendulumramp.png?raw=1" width="250" alt="pendulum on a ramp"/></figure>  

        6. Derive the Euler-Lagrange equation for the following Atwood machine (consider that$m_1 > m_2$, i.e., the pulley will rotate counter-clockwise, and that moving down is in the positive direction) (see solution at https://youtu.be/lVg8I23Khz4):  
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/atwood_machine.png?raw=1" width="125" alt="Atwood machine"/></figure>   

        7. Derive the Euler-Lagrange equation for the system below (see solution at https://youtu.be/gsi_0cVZ5-s):  
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/cart_pendulum_spring.png?raw=1" width="250" alt="cart_pendulum_spring"/></figure>  

        8. Consider a person hanging by their hands from a bar and this person oscillates freely, behaving like a rigid pendulum under the action of gravity. The person's mass is$m$, the distance from their center of gravity to the bar is$r$, their rotational inertia is$I$, a line that passes through the bar and the person's body forms an angle$\theta$with the vertical and the magnitude of the acceleration of gravity is$g$.  
          a. Derive the equation of motion for the person's body using the Lagrangian formalism.  
          b. If$m=100 kg$,$r=1 m$and$g=10 m/s^2$and the person takes$1 s$to perform a complete oscillation, calculate an estimate for the rotational inertia.  
          Tip: Use the approximation that for small angles$\sin\theta \approx \theta$and solve the differential equation.  
  
        9. Consider a mass-spring system under the action of gravity where the spring is attached to the ceiling and the mass in hanging in the vertical. The spring's proportionality constant is$k = 2 N/m$and its resting length is$\ell_0=1 m$. The mass attached to the spring is$2 kg$. Use$g = 10m/s^2$.  
          a. Derive the differential equation that describes the movement of mass.  
          b. Calculate the position of the mass over time considering that the spring is initially 2 m long and at rest.  
          c. Write a pseudocode to calculate the position of the mass by numerical calculation using Euler's method to solve the differential equation.  
  
        10. A widely used approach to study the interaction between the human body and the ground in locomotion is to model this interaction as a mass-spring-damper system with different quantities of these components. In an experiment to study this interaction during running, the vertical ground reaction force (GRF) was measured during the stance phase of a runner with a mass equal to$100 kg$and the magnitude of GRF versus time is shown in the figure.  <figure><img src="https://github.com/BMClab/BMC/blob/master/images/GRFv.png?raw=1" width="350" alt="GRFv"/></figure>  
          a. Draw a free-body diagram for the runner's center of gravity.   
          b. The maximum height of the runner in the aerial phase after the support phase shown in the graph above, knowing that the initial height and vertical speed (at$t=0 s$) of the runner (from his center of gravity) were$y_0 = 1 m$and$v_0 = –2  m/s$.   
          c. Draw the free-body diagram for a mechanical model of the corridor as a system consisting of a mass and spring and interaction with the ground.   
          d. Derive the equation of motion for this model obtained in item c.   
          e. Write a pseudocode to calculate the vertical trajectory from the differential equation obtained in item d by numerical calculation using Euler's method.

        11. Write computer programs (in Python!) to solve numerically the equations of motion from the problems above.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Hamilton WR (1834) [On a General Method in Dynamics](https://www.maths.tcd.ie/pub/HistMath/People/Hamilton/Dynamics/#GenMethod). Philosophical Transactions of the Royal Society, part II, 247-308.  
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
