import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Motion of a particle - Newtonian approach

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
        <h1>Contents<span class="tocSkip"></span></h1><br>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Python-setup" data-toc-modified-id="Python-setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Python setup</a></span></li><li><span><a href="#Study-of-motion" data-toc-modified-id="Study-of-motion-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Study of motion</a></span></li><li><span><a href="#The-development-of-the-laws-of-motion-of-bodies" data-toc-modified-id="The-development-of-the-laws-of-motion-of-bodies-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>The development of the laws of motion of bodies</a></span></li><li><span><a href="#Newton's-laws-of-motion" data-toc-modified-id="Newton's-laws-of-motion-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Newton's laws of motion</a></span></li><li><span><a href="#Steps-to-find-the-motion-of-a-particle" data-toc-modified-id="Steps-to-find-the-motion-of-a-particle-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Steps to find the motion of a particle</a></span><ul class="toc-item"><li><span><a href="#Example-1:-Ball-kicked-into-the-air" data-toc-modified-id="Example-1:-Ball-kicked-into-the-air-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Example 1: Ball kicked into the air</a></span><ul class="toc-item"><li><span><a href="#Analytical-solution" data-toc-modified-id="Analytical-solution-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>Analytical solution</a></span><ul class="toc-item"><li><span><a href="#Plot" data-toc-modified-id="Plot-5.1.1.1"><span class="toc-item-num">5.1.1.1&nbsp;&nbsp;</span>Plot</a></span></li></ul></li><li><span><a href="#Numerical-solution" data-toc-modified-id="Numerical-solution-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>Numerical solution</a></span><ul class="toc-item"><li><span><a href="#Plot" data-toc-modified-id="Plot-5.1.2.1"><span class="toc-item-num">5.1.2.1&nbsp;&nbsp;</span>Plot</a></span></li></ul></li></ul></li><li><span><a href="#Example-2:-Ball-kicked-into-the-air-considering-the-air-drag" data-toc-modified-id="Example-2:-Ball-kicked-into-the-air-considering-the-air-drag-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Example 2: Ball kicked into the air considering the air drag</a></span><ul class="toc-item"><li><span><a href="#Analytical-solution" data-toc-modified-id="Analytical-solution-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>Analytical solution</a></span><ul class="toc-item"><li><span><a href="#Plot" data-toc-modified-id="Plot-5.2.1.1"><span class="toc-item-num">5.2.1.1&nbsp;&nbsp;</span>Plot</a></span></li></ul></li><li><span><a href="#Numerical-solution" data-toc-modified-id="Numerical-solution-5.2.2"><span class="toc-item-num">5.2.2&nbsp;&nbsp;</span>Numerical solution</a></span><ul class="toc-item"><li><span><a href="#Plot" data-toc-modified-id="Plot-5.2.2.1"><span class="toc-item-num">5.2.2.1&nbsp;&nbsp;</span>Plot</a></span></li></ul></li></ul></li><li><span><a href="#Example-3:-Ball-kicked-into-the-air-considering-the-air-drag-proportional-to-square-of-speed" data-toc-modified-id="Example-3:-Ball-kicked-into-the-air-considering-the-air-drag-proportional-to-square-of-speed-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Example 3: Ball kicked into the air considering the air drag proportional to square of speed</a></span><ul class="toc-item"><li><span><a href="#All-numerical-solutions-plotted-together" data-toc-modified-id="All-numerical-solutions-plotted-together-5.3.1"><span class="toc-item-num">5.3.1&nbsp;&nbsp;</span>All numerical solutions plotted together</a></span></li></ul></li></ul></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-internet" data-toc-modified-id="Video-lectures-on-the-internet-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Video lectures on the internet</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Study of motion

        In Mechanics we are interested in the study of motion (including deformation) and forces (and the relation between them) of anything in nature.  

        As a good rule of thumb, we model the phenomenon of interest as simple as possible, with just enough complexity to understand the phenomenon. 

        For example, we could model a person jumping as a particle (the center of gravity, with no size) moving in one direction (the vertical) if all we want is to estimate the jump height and relate that to the external forces to the human body. So, mechanics of a particle might be all we need.  
        However, if the person jumps and performs a somersault, to understand this last part of the motion we have to model the human body as one of more objects which displaces and rotates in two or three dimensions. In this case, we would need what is called mechanics of rigid bodies.

        If, besides the gross motions of the segments of the body, we are interested in understanding the deformation in the the human body segments and tissues, now we would have to describe the mechanical behavior of the body (e.g., how it deforms) under the action of forces. In this case we would have to include some constitutive laws describing the mechanical properties of the body.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The development of the laws of motion of bodies  

        "The theoretical development of the laws of motion of bodies is a problem of such interest and importance that it has engaged the attention of all the most eminent mathematicians since the invention of dynamics as a mathematical science by Galileo, and especially since the wonderful extension which was given to that science by Newton."

        &#8212; Hamilton, 1834 (apud Taylor, 2005).  

        **Let's start with the study of the forces and motion in Mechanics looking at the motion of a particle using the Newtonian approach.**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Newton's laws of motion

        The Newton's laws of motion describe the relationship between the forces acting on a body and the resultant linear motion due to those forces:

        - **First law**: An object will remain at rest or in uniform motion in a straight line unless an external force acts on the body.
        - **Second law**: The acceleration of an object is directly proportional to the net force acting on the object and inversely proportional to the mass of the object:$\vec{\bf{F}} = m \vec{\bf{a}}$.
        - **Third law**: Whenever an object exerts a force$\vec{\bf{F}}_1$(action) on a second object, this second object simultaneously exerts a force$\vec{\bf{F}}_2$on the first object with the same magnitude but opposite direction (reaction):$\vec{\bf{F}}_2 = −\vec{\bf{F}}_1.$These three statements are astonishing in their simplicity and how much of knowledge they empower.   
        Isaac Newton was born in 1943 and his works that resulted in these equations and other discoveries were mostly done in the years of 1666 and 1667, when he was only 24 years old!  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Steps to find the motion of a particle

        1. Draw a free body diagram of the particle. Draw all the forces being applied to the particle.  
        2. Write the expression of each force applied to the particle. For external forces (for example gravity and air friction) write the constitutive laws of the phenomena.  
        3. Write the Newton's second Law$\vec{\bf{F}} = m \vec{\bf{a}}$, where$\vec{\bf{F}}$is the sum of all forces applied to the particle and$\vec{\bf{a}}$is the particle acceleration.  
        4. Separate the equation into the 3 cartesian components (or 2 components if the movement is bidimensional).  
        5. Solve the differential equations
         1. If possible, solve the differential equations analytically.  
         2. If not possible to solve the differential equations analytically, separate each equation into 2 first order differential equations and use some numerical method (e.g. Euler, Runge-Kutta) to solve the first order differential equations with the aid of a computer.   
        6. Use the solution to interpret the situation, or to find some error on your approach.  

        **Later, we will study in details how to draw a free-body diagram.**  
        Let's see now some examples on how to find the motion of a particle
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1: Ball kicked into the air

        #### Analytical solution

        A football ball is kicked with an angle of 30 degrees with the ground (horizontal).  
        The mass of the ball is 0.43 kg. The initial speed of the ball is 20 m/s and the initial height is 0 m. Consider the gravitational acceleration as 9.81$m/s^2$.  
        Find the motion of the ball. 

        Solution:  
        We know that:  
        <span class="notranslate">$x_0 = 0 m \\ y_0 = 0 m$</span>
        As the angle of the initial velocity of the ball with the ground is 30 degrees:  
        <span class="notranslate">$v_{x0} = 20 \cos(30^\circ) = 20\frac{\sqrt{3}}{2} = 10\sqrt{3} m/s \\
        v_{y0} = 20 \sin(30^\circ) = 20 \frac{1}{2} = 10 m/s$</span>
        The free-body diagram of the ball is depicted below:
   
        <figure><center><img src="../images/ballGrav.png" alt="free-body diagram of a ball" width="500"/><figcaption><i>Figure. Free-body diagram of a ball under the influence of gravity.</i></figcaption></center></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The only force acting on the ball is the gravitational force:
        <span class="notranslate">$\vec{\bf{F}}_g = -mg \; \hat{\bf{j}}$</span>
        So, we apply the Newton's second law:
        <span class="notranslate">$\vec{\bf{F}}_g = m \frac{d^2\vec{\bf{r}}}{dt^2} \quad \rightarrow \quad - mg \; \hat{\bf{j}} = m \frac{d^2\vec{\bf{r}}}{dt^2} \quad \rightarrow \quad - g \; \hat{\bf{j}} = \frac{d^2\vec{\bf{r}}}{dt^2}$</span>
        Now, we can separate the equation in two components (x and y):
        <span class="notranslate">$0 = \frac{d^2x}{dt^2}$</span>
        and
        <span class="notranslate">$-g = \frac{d^2y}{dt^2}$</span>
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    # '%matplotlib inline' command supported automatically in marimo
    sns.set_context('notebook', font_scale=1.2)
    return np, plt


@app.cell
def _(np):
    m   = 0.43           # [kg]
    x0  = 0              # [m]
    y0  = 0              # [m]
    vx0 = 10*np.sqrt(3)  # [m/s]
    vy0 = 10             # [m/s]
    g   = 9.81           # [m/s^2]
    return g, m, vx0, vy0, x0, y0


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        These equations can be easily solved  by integrating both sides of each equation:
        <span class="notranslate">$0 = \frac{d^2x}{dt^2} \quad \rightarrow v_{x0} = \frac{dx}{dt} \quad \rightarrow \quad v_{x0}t + x_{0} = x(t)$</span>
        and
        <span class="notranslate">$-g = \frac{d^2y}{dt^2} \quad \rightarrow - gt + v_{y0} = \frac{dy}{dt} \quad \rightarrow \quad -\frac{g}{2}t^2 +v_{y0}t + y_{0} = y(t)$</span>
        So the trajectory of the ball along time is:
        <span class="notranslate">$x(t) = v_{x0}t$</span>
        and 
        <span class="notranslate">$y(t) =  -\frac{g}{2} t^2 + v_{y0}t$</span>
        or 
        <span class="notranslate">$\vec{\bf{r}}(t) =  v_{x0}t \; \hat{\bf{i}} + \left(-\frac{g}{2} t^2 + v_{y0}t \right) \; \hat{\bf{j}}$</span>
        """
    )
    return


@app.cell
def _(g, np, plt, vx0, vy0):
    dt = 0.001
    _t = np.arange(0, 2.05, dt)
    x1a = vx0 * _t
    y1a = -g / 2 * _t ** 2 + vy0 * _t
    plt.figure(figsize=(8, 4))
    plt.plot(x1a, y1a, lw=4)
    plt.xlim(0, 36)
    plt.ylim(0, 6)
    plt.title('Ball trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return (dt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Numerical solution

        We can solve this problem numerically and compare both solutions.

        We start from the differential equations of each coordinate, and then break them into two first-order differential equations:
        <span class="notranslate">$0 = \frac{d^2x}{dt^2}$</span>
        and
        <span class="notranslate">$- g = \frac{d^2y}{dt^2}$</span>
        The first equation can be broken as:
        <span class="notranslate">$\frac{dv_x}{dt} = 0$</span>
        <span class="notranslate">$\frac{dx}{dt} = v_x(t)$</span>
        And the second equation can be broken as:
        <span class="notranslate">$\frac{dv_y}{dt} = -g$</span>
        <span class="notranslate">$\frac{dy}{dt} = v_y(t)$</span>
        You can use any numerical integration method you want (Euler, Runge-Kutta, etc), but here we will use the Euler method. Let's see the solution for the$x(t)$variable.

        The derivative of$x(t)$is given by:
        <span class="notranslate">$\frac{dx}{dt} = \lim\limits_{\Delta t \rightarrow 0} \frac{x(t+\Delta t) - x(t)}{\Delta t}$</span>
        Whcih can be approximated by: 
        <span class="notranslate">$\frac{dx}{dt} \approx \frac{x(t+\Delta t) - x(t)}{\Delta t} \quad \rightarrow \quad x(t+\Delta t) \approx x(t) + \Delta t \frac{dx}{dt}$</span>
        So, with the initial conditions of all the variables, we can apply the equation above to find the values of the variables along time (for a revision of Ordinary Differential Equations, see the notebook [Ordinary Differential Equation](https://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/OrdinaryDifferentialEquation.ipynb)).  
        In the cell below, we apply the Euler method for the four first-order differential equations.
        """
    )
    return


@app.cell
def _(dt, g, np, vx0, vy0, x0, y0):
    _x = x0
    _y = y0
    _vx = vx0
    _vy = vy0
    r = np.array([_x, _y])
    while _y >= 0:
        _dxdt = _vx
        _x = _x + dt * _dxdt
        _dydt = _vy
        _y = _y + dt * _dydt
        _dvxdt = 0
        _vx = _vx + dt * _dvxdt
        _dvydt = -g
        _vy = _vy + dt * _dvydt
        r = np.vstack((r, np.array([_x, _y])))
    return (r,)


@app.cell
def _(plt, r):
    plt.figure(figsize=(8, 4))
    x1n = r[:, 0]
    y1n = r[:, 1]
    plt.plot(x1n, y1n, lw=4)
    plt.xlim(0, 36)
    plt.ylim(0, 6)
    plt.title('Ball trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return x1n, y1n


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2: Ball kicked into the air considering the air drag

        Now, besides the gravity, we consider the drag due to the air resistance ($b$= 0.006 Ns/m). First we will consider the drag force  proportional to the speed and opposite direction to the velocity vector.
       
        <figure><center><img src="../images/ballGravLinearRes.png" alt="free-body diagram of the ball" width="600"/><figcaption><i>Figure. Free-body diagram of a ball under the influence of gravity and drag.</i></figcaption></center></figure>

        So the forces being applied on the ball are:
        <span class="notranslate">$\vec{\bf{F}} = -mg \; \hat{\bf{j}} - b\vec{\bf{v}} = -mg \; \hat{\bf{j}} - b\frac{d\vec{\bf{r}}}{dt} = -mg \;  \hat{\bf{j}} - b\left(\frac{dx}{dt} \; \hat{\bf{i}}+\frac{dy}{dt} \; \hat{\bf{j}}\right) = - b\frac{dx}{dt} \; \hat{\bf{i}} - \left(mg + b\frac{dy}{dt}\right) \; \hat{\bf{j}}$</span>
        Writing down the Newton's second law:
        <span class="notranslate">$\vec{\bf{F}} = m \frac{d^2\vec{\bf{r}}}{dt^2} \quad \rightarrow \quad - b\frac{dx}{dt} \; \hat{\bf{i}} - \left(mg + b\frac{dy}{dt}\right) \; \hat{\bf{j}} = m\left(\frac{d^2x}{dt^2} \; \hat{\bf{i}}+\frac{d^2y}{dt^2} \; \hat{\bf{j}}\right)$</span>
        Now, we can separate into one equation for each coordinate:
        <span class="notranslate">$- b\frac{dx}{dt} = m\frac{d^2x}{dt^2} \quad \rightarrow \quad \frac{d^2x}{dt^2} = -\frac{b}{m} \frac{dx}{dt}$</span>
        <span class="notranslate">$-mg - b\frac{dy}{dt} = m\frac{d^2y}{dt^2} \quad \rightarrow \quad \frac{d^2y}{dt^2} = -\frac{b}{m}\frac{dy}{dt} - g$</span>
        We can solve these equations analytically, for example, by using Laplace Transform or classical methods to solve linear differential equations.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Analytical solution

        The solution of a linear differential equation can be found by finding the natural (or homogeneous) solution and the forced (or non-homogeneous) solution and then adding both solutions.

        First, we solve the first differential equation ($x$coordinate). The characteristic polynomial of the equation is:

        <span class="notranslate">$\lambda^2 + \frac{b}{m}\lambda = 0$</span>

        The roots of this equation are$\lambda = 0$and$\lambda = -\frac{b}{m}$, and consequently, its natural modes are:

        <span class="notranslate">$x_{n_1}(t) = Ae^{0t} = A \\
        x_{n_2}(t) = B e^{-\frac{b}{m}t}$</span>

        As there is no external forces in the$x$direction, there is no forced solution. So, the motion of the ball in the$x$coordinate is:

        <span class="notranslate">$x(t) = A + Be^{-\frac{b}{m}t}$</span>

        To find the values of the$A$and$B$constants, we must use the initial conditions$x(0)$and$v_x(0)$.

        <span class="notranslate">$x(0) = 0 = A + B$</span>
        <span class="notranslate">$v_x(0) = v_{x0} = \frac{dx(0)}{dt} = -\frac{Bb}{m}e^{-\frac{b}{m}0} \quad \rightarrow \quad B = -\frac{v_{x0}m}{b} \quad \rightarrow \quad A = \frac{v_{x0}m}{b}$</span>
        So:
        <span class="notranslate">$x(t) = \frac{v_{x0}m}{b} - \frac{v_{x0}m}{b}e^{-\frac{b}{m}t} = \frac{v_{x0}m}{b}\left(1-e^{-\frac{b}{m}t} \right)$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now, we solve the differential equation for the$y$coordinate.  
        First, we find the natural solution, which comprises the solution from the equation without the gravitational force (the force indepedent of y(t) and its derivatives), in this case:
        <span class="notranslate">$\frac{d^2y}{dt^2} = -\frac{b}{m}\frac{dy}{dt}$</span>
        The solution of this equation is the same from the$x$coordinate:
        <span class="notranslate">$y_n(t) = A + Be^{-\frac{b}{m}t}$</span>
        The forced solution (including the gravitational force, which is constant) happens when every derivative, with the exception of the derivative with the lowest order (in this case order 1 but it could be order 0, i.e. no derivative) goes to zero. 
        <span class="notranslate">$\frac{d^2y}{dt^2} \quad = \quad 0 \quad = \quad -\frac{b}{m}\frac{dy_f}{dt} - g \quad \rightarrow \quad \frac{dy_f}{dt} = -\frac{mg}{b} \quad \rightarrow \quad y_f(t) = -\frac{mg}{b}t$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The complete solution of the motion of the ball in the$y$coordinate is the sum of the natural and forced solutions:

        <span class="notranslate">$y(t) = A + Be^{-\frac{b}{m}t} - \frac{mg}{b}t$</span>

        To find the values of the constants A and B, we must use the initial conditions$y(0)$and$v_y(0)$.

        <span class="notranslate">$y(0) = 0 = A + B$</span>

        <span class="notranslate">$v_y(0) = v_{y0} = \frac{dy(0)}{dt} = - \frac{Bb}{m} - \frac{mg}{b} \\
        B = -\frac{m^2g}{b^2} -\frac{v_{y0}m}{b} \\
        A = +\frac{m^2g}{b^2} + \frac{v_{y0}m}{b}$</span>

        So, the motion of the ball in the y coordinate is:

        <span class="notranslate">$y(t) = \left(\frac{m^2g}{b^2} + \frac{v_{y0}m}{b}\right) - \left(\frac{m^2g}{b^2} + \frac{v_{y0}m}{b}\right)e^{-\frac{b}{m}t} -\frac{mg}{b}t \\
        y(t) = \left(\frac{m^2g}{b^2} + \frac{v_{y0}m}{b}\right)\left(1 - e^{-\frac{b}{m}t}\right) - \frac{mg}{b}t$</span>
        """
    )
    return


@app.cell
def _(g, m, np, plt, vx0, vy0):
    b = 0.006
    _t = np.arange(0, 2.05, 0.01)
    x2a = vx0 * m / b * (1 - np.exp(-b / m * _t))
    y2a = (vy0 * m / b + g * m ** 2 / b ** 2) * (1 - np.exp(-b / m * _t)) - g * m / b * _t
    plt.figure(figsize=(8, 4))
    plt.plot(x2a, y2a, lw=4)
    plt.xlim(0, 36)
    plt.ylim(0, 6)
    plt.title('Ball trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return (b,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Numerical solution

        Now, we will solve the same situation using a numerical method (Euler method).  
        We start from the equations previously found for each coordinate:

        <span class="notranslate">$\frac{d^2x}{dt^2} = -\frac{b}{m} \frac{dx}{dt}$</span>
        <span class="notranslate">$\frac{d^2y}{dt^2} = -\frac{b}{m}\frac{dy}{dt} - g$</span>

        We can separate each equation into two first order equations and apply the Euler method:  
        <br>

        <span class="notranslate">$\frac{dv_x}{dt} = -\frac{b}{m} v_x$</span>
        <span class="notranslate">$\frac{dx}{dt} = v_x$</span>
        <span class="notranslate">$\frac{dv_y}{dt} = -\frac{b}{m}v_y - g$</span>
        <span class="notranslate">$\frac{dy}{dt} = v_y$</span>
        """
    )
    return


@app.cell
def _(b, dt, g, m, np, vx0, vy0, x0, y0):
    _x = x0
    _y = y0
    _vx = vx0
    _vy = vy0
    r_1 = np.array([_x, _y])
    while _y >= 0:
        _dxdt = _vx
        _x = _x + dt * _dxdt
        _dydt = _vy
        _y = _y + dt * _dydt
        _dvxdt = -b / m * _vx
        _vx = _vx + dt * _dvxdt
        _dvydt = -g - b / m * _vy
        _vy = _vy + dt * _dvydt
        r_1 = np.vstack((r_1, np.array([_x, _y])))
    return (r_1,)


@app.cell
def _(plt, r_1):
    plt.figure(figsize=(8, 4))
    x2n = r_1[:, 0]
    y2n = r_1[:, 1]
    plt.plot(x2n, y2n, lw=4)
    plt.xlim(0, 36)
    plt.ylim(0, 6)
    plt.title('Ball trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return x2n, y2n


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 3: Ball kicked into the air considering the air drag proportional to square of speed

        Now, we will consider the drag force due to the air resistance proportional to the square of speed and still in the opposite direction of the velocity vector. 
           
        <figure><center><img src="../images/ballGravSquareRes.png" alt="free-body diagram of the ball" width="600"/><figcaption><i>Figure. Free-body diagram of a ball under the influence of gravity and drag proportional to the square of speed.</i></figcaption></center></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        So the forces being applied on the ball are (for a revision of Time-varying frames and the meaning of the$\hat{\bf{e_t}}$, see [Time-varying frames notebook](https://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/Time-varying%20frames.ipynb)):
        <span class="notranslate">$\vec{\bf{F}} = -mg \; \hat{\bf{j}} - bv^2\hat{\bf{e_t}} \\
        \vec{\bf{F}} = -mg \; \hat{\bf{j}} - b (v_x^2+v_y^2) \frac{v_x \; \hat{\bf{i}} + v_y \; \hat{\bf{j}}}{\sqrt{v_x^2+v_y^2}} \\
        \vec{\bf{F}} = -mg \; \hat{\bf{j}} - b \sqrt{v_x^2+v_y^2} \,(v_x \; \hat{\bf{i}}+v_y \; \hat{\bf{j}}) \\
        \vec{\bf{F}} = -mg \; \hat{\bf{j}} - b \sqrt{\left(\frac{dx}{dt} \right)^2+\left(\frac{dy}{dt} \right)^2} \,\left(\frac{dx}{dt} \hat{\bf{i}} + \frac{dy}{dt} \; \hat{\bf{j}}\right)$</span>
        Writing down the Newton's second law:
        <span class="notranslate">$\vec{\bf{F}} = m \frac{d^2\vec{\bf{r}}}{dt^2} \\
        -mg \; \hat{\bf{j}} - b \sqrt{\left(\frac{dx}{dt} \right)^2+\left(\frac{dy}{dt} \right)^2} \,\left(\frac{dx}{dt} \hat{\bf{i}}+\frac{dy}{dt}\hat{\bf{j}}\right) = m\left(\frac{d^2x}{dt^2}\hat{\bf{i}}+\frac{d^2y}{dt^2}\hat{\bf{j}}\right)$</span>
        Now, we can separate into one equation for each coordinate:
        <span class="notranslate">$- b \sqrt{\left(\frac{dx}{dt} \right)^2+\left(\frac{dy}{dt} \right)^2} \,\frac{dx}{dt} = m\frac{d^2x}{dt^2} \quad \rightarrow \\
        \frac{d^2x}{dt^2} = - \frac{b}{m} \sqrt{\left(\frac{dx}{dt} \right)^2+\left(\frac{dy}{dt} \right)^2} \,\frac{dx}{dt}$</span>
        <span class="notranslate">$-mg - b \sqrt{\left(\frac{dx}{dt} \right)^2+\left(\frac{dy}{dt} \right)^2} \,\frac{dy}{dt} = m\frac{d^2y}{dt^2} \quad \rightarrow \\
        \frac{d^2y}{dt^2} = - \frac{b}{m} \sqrt{\left(\frac{dx}{dt} \right)^2 + \left(\frac{dy}{dt} \right)^2} \,\frac{dy}{dt} -g$</span>
        These equations are very difficult to solve analytically, but they can be easily  solved by using numerical methods. Below we  will use the same numerical method (Euler method) to solve these equations.  
        For that, again we must break each equation into two first-order differential equations:
        <span class="notranslate">$\frac{dv_x}{dt} = - \frac{b}{m} \sqrt{v_x^2+v_y^2} \,v_x$</span>
        <span class="notranslate">$\frac{dx}{dt} = v_x$</span>
        <span class="notranslate">$\frac{dv_y}{dt} = - \frac{b}{m} \sqrt{v_x^2+v_y^2} \,v_y -g$</span>
        <span class="notranslate">$\frac{dy}{dt} = v_y$</span>
        Now, we can apply the Euler method to find a solution. 
        """
    )
    return


@app.cell
def _(b, dt, g, m, np, vx0, vy0, x0, y0):
    _x = x0
    _y = y0
    _vx = vx0
    _vy = vy0
    r_2 = np.array([_x, _y])
    while _y >= 0:
        _dxdt = _vx
        _x = _x + dt * _dxdt
        _dydt = _vy
        _y = _y + dt * _dydt
        _dvxdt = -b / m * np.sqrt(_vx ** 2 + _vy ** 2) * _vx
        _vx = _vx + dt * _dvxdt
        _dvydt = -b / m * np.sqrt(_vx ** 2 + _vy ** 2) * _vy - g
        _vy = _vy + dt * _dvydt
        r_2 = np.vstack((r_2, np.array([_x, _y])))
    return (r_2,)


@app.cell
def _(plt, r_2):
    plt.figure(figsize=(8, 4))
    x3n = r_2[:, 0]
    y3n = r_2[:, 1]
    plt.plot(x3n, y3n, lw=4)
    plt.xlim(0, 36)
    plt.ylim(0, 6)
    plt.title('Ball trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return x3n, y3n


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### All numerical solutions plotted together
        """
    )
    return


@app.cell
def _(plt, x1n, x2n, x3n, y1n, y2n, y3n):
    plt.figure(figsize=(8, 4))
    plt.plot(x1n, y1n, lw=4, c='g', label='$g$')
    plt.plot(x2n, y2n, lw=4, c='b', label='$g+bv$')
    plt.plot(x3n, y3n, lw=4, c='r', label='$g+bv^2$')
    plt.xlim(0, 36)
    plt.ylim(0, 6)
    plt.title('Ball trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best', frameon=False)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

        - Read the chapter 0 of the [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html) about mechanics  
        - Read pages 478-494 of the chapter 10 of the [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html) about 1D dynamics  
        - Read the chapter 13 of the [Hibbeler's book](https://drive.google.com/file/d/1sDLluWCiBCog2C11_Iu1fjv-BtfVUxBU/view) (available in the Classroom)  
        - See [What is the fastest possible volleyball serve?](https://uio-ccse.github.io/computational-essay-showroom/essays/exampleessays/volleyball/Volleyball.html) for a nice investigation about the ball movement during a volleyball serve
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the internet

        - Khan Academy: [Forces and Newton's laws of motion](https://www.khanacademy.org/science/ap-physics-1/ap-forces-newtons-laws) (Em português: [Forças e as Leis do Movimento de Newton](https://pt.khanacademy.org/science/physics/forces-newtons-laws)).  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. (Example 13.1, Hibbeler's book) A$50 kg$crate rests on a horizontal surface for which the coefficient of kinetic friction is$\mu_k = 0.3$. If the crate is subjected to a$400 N$towing force applied upward with a$30^o$angle w.r.t. the horizontal, determine the velocity of the crate after$3 s$starting from rest.  

        2. (Example 13.2, Hibbeler's book) A 10 kg projectile is fired vertically upward from the ground, with an initial velocity of 50$m/s$. Determine the maximum height to which it will travel if (a) atmospheric resistance is neglected; and (b) atmospheric resistance is measured as$F_D=0.01v^2$$N$, where$v$is the speed of the projectile at any instant, measured in$m/s$.  

        3. (Exercise 13.6, Hibbeler's book) A person pushes on a$60 kg$crate with a force$F$. The force is always directed down at$30^o$from the horizontal as shown, and its magnitude is increased until the crate begins to slide. Determine the crate’s initial acceleration if the coefficient of static friction is$\mu_s=0.6$and the coefficient of kinetic friction is$\mu_k=0.3$.  

        4. (Exercise 13.46, Hibbeler's book) The parachutist of mass$m$is falling with a velocity of$v_0$at the instant she opens the parachute. If air resistance is$F_D = Cv^2$, show that her maximum velocity (terminal velocity) during the descent is$v_{max}=\sqrt{{mg}/{C}}$. Solve the problem numerically considering$m=100 kg$,$g=10 m/s^2$, and$C=10 kg/m$. Plot a simulation of this numerical solution and show that indeed the parachutist approaches the terminal velocity.

        5. Consider a block with mass of$1 kg$attached to a spring hanging from a ceiling (the spring constant$k = 100 N/m$). At$t = 0 s$, the spring is stretched by$0.1 m$from the equilibrium position of the block + spring system and then it's released (the initial velocity is not specified). Find the motion of the block.

        6. Solve exercises 12.1.16, 12.1.19, 12.1.24, 12.1.29, 12.1.30, 12.1.31(a, b, d) and 12.1.32 from Ruina and Pratap's book (2019).  

        7. (https://youtu.be/N6IhkTjWrd4) A$15 kg$box rests on a frictionless horizontal surface attached to a$5 kg$box as shown below.  
          a. What is the acceleration of the system?  
          b. What is the tension in the rope?  
          c. Now consider a coefficient of kinetic friction of$0.25$between the horizontal surface and the box. What are the acceleration of the system and the tension in the rope? 
  
        <figure><center><img src="../images/boxes_pulley_rope.png" alt="free-body diagram of the ball" width="300"/></center></figure>

        8. Consider a moving particle in a fluid with viscosity proportional to the cube of the velocity. This particle moves only vertically and never reaches the ground. Consider the magnitude of the acceleration due to gravity as$10 m/s^2$.  
           a. Draw the free body diagram.  
           b. What is the differential equation that describes the particle's motion?  
           c. If the viscosity coefficient is$5 Ns^3/m^3$, what is the maximum speed of the particle?


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Hibbeler RC (2010) [Engineering Mechanics Dynamics](https://drive.google.com/file/d/1sDLluWCiBCog2C11_Iu1fjv-BtfVUxBU/view). 12th Edition. Pearson Prentice Hall.
        - Ruina A, Rudra P (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.  
        - Taylor JR (2005) [Classical Mechanics](https://books.google.com.br/books?id=P1kCtNr-pJsC). University Science Books.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
