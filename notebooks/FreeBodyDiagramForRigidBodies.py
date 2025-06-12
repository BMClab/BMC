import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Free Body Diagram for Rigid Bodies

        > Renato Naville Watanabe  
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
        <div class="toc"><ul class="toc-item"><li><span><a href="#Python-setup" data-toc-modified-id="Python-setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Python setup</a></span></li><li><span><a href="#Equivalent-systems" data-toc-modified-id="Equivalent-systems-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Equivalent systems</a></span></li><li><span><a href="#Steps-to-draw-a-free-body-diagram" data-toc-modified-id="Steps-to-draw-a-free-body-diagram-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Steps to draw a free-body diagram</a></span></li><li><span><a href="#Examples" data-toc-modified-id="Examples-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Examples</a></span><ul class="toc-item"><li><span><a href="#Horizontal-fixed-bar" data-toc-modified-id="Horizontal-fixed-bar-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Horizontal fixed bar</a></span><ul class="toc-item"><li><span><a href="#Free-body-diagram" data-toc-modified-id="Free-body-diagram-4.1.1"><span class="toc-item-num">4.1.1&nbsp;&nbsp;</span>Free-body diagram</a></span></li><li><span><a href="#Sum-of-the-forces-and-moments" data-toc-modified-id="Sum-of-the-forces-and-moments-4.1.2"><span class="toc-item-num">4.1.2&nbsp;&nbsp;</span>Sum of the forces and moments</a></span></li><li><span><a href="#Newton-Euler-laws" data-toc-modified-id="Newton-Euler-laws-4.1.3"><span class="toc-item-num">341.3&nbsp;&nbsp;</span>Newton-Euler laws</a></span></li></ul></li><li><span><a href="#Rotating-ball-with-drag-force" data-toc-modified-id="Rotating-ball-with-drag-force-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Rotating ball with drag force</a></span><ul class="toc-item"><li><span><a href="#Free-body-diagram" data-toc-modified-id="Free-body-diagram-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>Free-body diagram</a></span></li><li><span><a href="#Sum-of-forces-and-moments" data-toc-modified-id="Sum-of-forces-and-moments-4.2.2"><span class="toc-item-num">4.2.2&nbsp;&nbsp;</span>Sum of forces and moments</a></span></li><li><span><a href="#Newton-Euler-laws" data-toc-modified-id="Newton-Euler-laws-4.2.3"><span class="toc-item-num">4.2.3&nbsp;&nbsp;</span>Newton-Euler laws</a></span></li><li><span><a href="#Equations-of-motion" data-toc-modified-id="Equations-of-motion-4.2.4"><span class="toc-item-num">4.2.4&nbsp;&nbsp;</span>Equations of motion</a></span></li><li><span><a href="#Numerical-solution-of-the-equations" data-toc-modified-id="Numerical-solution-of-the-equations-4.2.5"><span class="toc-item-num">4.2.5&nbsp;&nbsp;</span>Numerical solution of the equations</a></span></li></ul></li><li><span><a href="#Pendulum" data-toc-modified-id="Pendulum-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Pendulum</a></span><ul class="toc-item"><li><span><a href="#Free-body-diagram" data-toc-modified-id="Free-body-diagram-4.3.1"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>Free-body diagram</a></span></li><li><span><a href="#Sum-of-the-forces-and-moments" data-toc-modified-id="Sum-of-the-forces-and-moments-4.3.2"><span class="toc-item-num">4.3.2&nbsp;&nbsp;</span>Sum of the forces and moments</a></span></li><li><span><a href="#Second-Newton-Euler-law" data-toc-modified-id="Second-Newton-Euler-law-4.3.3"><span class="toc-item-num">4.3.3&nbsp;&nbsp;</span>Second Newton-Euler law</a></span></li><li><span><a href="#Kinematics" data-toc-modified-id="Kinematics-4.3.4"><span class="toc-item-num">4.3.4&nbsp;&nbsp;</span>Kinematics</a></span></li><li><span><a href="#Sum-of-the-moments" data-toc-modified-id="Sum-of-the-moments-4.3.5"><span class="toc-item-num">4.3.5&nbsp;&nbsp;</span>Sum of the moments</a></span></li><li><span><a href="#Derivative-of-the-angular-momentum" data-toc-modified-id="Derivative-of-the-angular-momentum-4.3.6"><span class="toc-item-num">4.3.6&nbsp;&nbsp;</span>Derivative of the angular momentum</a></span></li><li><span><a href="#The-equation-of-motion" data-toc-modified-id="The-equation-of-motion-4.3.7"><span class="toc-item-num">4.3.7&nbsp;&nbsp;</span>The equation of motion</a></span></li></ul></li><li><span><a href="#Inverted-Pendulum" data-toc-modified-id="Inverted-Pendulum-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Inverted Pendulum</a></span><ul class="toc-item"><li><span><a href="#Free-body-diagram" data-toc-modified-id="Free-body-diagram-4.4.1"><span class="toc-item-num">4.4.1&nbsp;&nbsp;</span>Free-body diagram</a></span></li><li><span><a href="#Sum-of-the-forces-and-moments" data-toc-modified-id="Sum-of-the-forces-and-moments-4.4.2"><span class="toc-item-num">4.4.2&nbsp;&nbsp;</span>Sum of the forces and moments</a></span></li><li><span><a href="#Second-Newton-Euler-law" data-toc-modified-id="Second-Newton-Euler-law-4.4.3"><span class="toc-item-num">4.4.3&nbsp;&nbsp;</span>Second Newton-Euler law</a></span></li><li><span><a href="#Kinematics" data-toc-modified-id="Kinematics-4.4.4"><span class="toc-item-num">4.4.4&nbsp;&nbsp;</span>Kinematics</a></span></li><li><span><a href="#Sum-of-the-moments" data-toc-modified-id="Sum-of-the-moments-4.4.5"><span class="toc-item-num">4.4.5&nbsp;&nbsp;</span>Sum of the moments</a></span></li><li><span><a href="#Derivative-of-the-angular-momentum" data-toc-modified-id="Derivative-of-the-angular-momentum-4.4.6"><span class="toc-item-num">4.4.6&nbsp;&nbsp;</span>Derivative of the angular momentum</a></span></li><li><span><a href="#The-equation-of-motion" data-toc-modified-id="The-equation-of-motion-4.4.7"><span class="toc-item-num">4.4.7&nbsp;&nbsp;</span>The equation of motion</a></span></li></ul></li><li><span><a href="#Human-quiet-standing" data-toc-modified-id="Human-quiet-standing-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Human quiet standing</a></span><ul class="toc-item"><li><span><a href="#Free-body-diagram" data-toc-modified-id="Free-body-diagram-4.5.1"><span class="toc-item-num">4.5.1&nbsp;&nbsp;</span>Free-body diagram</a></span></li><li><span><a href="#Sum-of-the-moments" data-toc-modified-id="Sum-of-the-moments-4.5.2"><span class="toc-item-num">4.5.2&nbsp;&nbsp;</span>Sum of the moments</a></span></li><li><span><a href="#Second-Newton-Euler-law" data-toc-modified-id="Second-Newton-Euler-law-4.5.3"><span class="toc-item-num">4.5.3&nbsp;&nbsp;</span>Second Newton-Euler law</a></span></li><li><span><a href="#Kinematics" data-toc-modified-id="Kinematics-4.5.4"><span class="toc-item-num">4.5.4&nbsp;&nbsp;</span>Kinematics</a></span></li><li><span><a href="#Sum-of-the-moments" data-toc-modified-id="Sum-of-the-moments-4.5.5"><span class="toc-item-num">4.5.5&nbsp;&nbsp;</span>Sum of the moments</a></span></li></ul></li><li><span><a href="#Derivative-of-the-angular-momentum" data-toc-modified-id="Derivative-of-the-angular-momentum-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Derivative of the angular momentum</a></span><ul class="toc-item"><li><span><a href="#The-equation-of-motion" data-toc-modified-id="The-equation-of-motion-4.6.1"><span class="toc-item-num">4.6.1&nbsp;&nbsp;</span>The equation of motion</a></span></li></ul></li><li><span><a href="#Force-platform-(position-of-the-center-of-pressure)" data-toc-modified-id="Force-platform-(position-of-the-center-of-pressure)-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Force platform (position of the center of pressure)</a></span><ul class="toc-item"><li><span><a href="#Free-body-diagram" data-toc-modified-id="Free-body-diagram-4.7.1"><span class="toc-item-num">4.7.1&nbsp;&nbsp;</span>Free-body diagram</a></span></li></ul></li><li><span><a href="#Standing-dummy-(from-Ruina-and-Pratap-book)" data-toc-modified-id="Standing-dummy-(from-Ruina-and-Pratap-book)-4.8"><span class="toc-item-num">4.8&nbsp;&nbsp;</span>Standing dummy (from Ruina and Pratap book)</a></span><ul class="toc-item"><li><span><a href="#Free-body-diagram" data-toc-modified-id="Free-body-diagram-4.8.1"><span class="toc-item-num">4.8.1&nbsp;&nbsp;</span>Free-body diagram</a></span></li><li><span><a href="#Kinematics" data-toc-modified-id="Kinematics-4.8.2"><span class="toc-item-num">4.8.2&nbsp;&nbsp;</span>Kinematics</a></span></li><li><span><a href="#Sum-of-the-moments" data-toc-modified-id="Sum-of-the-moments-4.8.3"><span class="toc-item-num">4.8.3&nbsp;&nbsp;</span>Sum of the moments</a></span></li><li><span><a href="#Derivative-of-the-angular-momentum" data-toc-modified-id="Derivative-of-the-angular-momentum-4.8.4"><span class="toc-item-num">4.8.4&nbsp;&nbsp;</span>Derivative of the angular momentum</a></span></li><li><span><a href="#The-equation-of-motion" data-toc-modified-id="The-equation-of-motion-4.8.5"><span class="toc-item-num">4.8.5&nbsp;&nbsp;</span>The equation of motion</a></span></li></ul></li><li><span><a href="#Double-pendulum-with-actuators" data-toc-modified-id="Double-pendulum-with-actuators-4.9"><span class="toc-item-num">4.9&nbsp;&nbsp;</span>Double pendulum with actuators</a></span><ul class="toc-item"><li><span><a href="#Free-body-diagram" data-toc-modified-id="Free-body-diagram-4.9.1"><span class="toc-item-num">4.9.1&nbsp;&nbsp;</span>Free-body diagram</a></span></li><li><span><a href="#Definition-of-versors" data-toc-modified-id="Definition-of-versors-4.9.2"><span class="toc-item-num">4.9.2&nbsp;&nbsp;</span>Definition of versors</a></span></li><li><span><a href="#Kinematics" data-toc-modified-id="Kinematics-4.9.3"><span class="toc-item-num">4.9.3&nbsp;&nbsp;</span>Kinematics</a></span></li><li><span><a href="#Sum-of-the-moments-of-the-lower-bar-around-O2" data-toc-modified-id="Sum-of-the-moments-of-the-lower-bar-around-O2-4.9.4"><span class="toc-item-num">4.9.4&nbsp;&nbsp;</span>Sum of the moments of the lower bar around O2</a></span></li><li><span><a href="#Derivative-of-angular-momentum-of-the-lower-bar-around-O2" data-toc-modified-id="Derivative-of-angular-momentum-of-the-lower-bar-around-O2-4.9.5"><span class="toc-item-num">4.9.5&nbsp;&nbsp;</span>Derivative of angular momentum of the lower bar around O2</a></span></li><li><span><a href="#Sum-of-moments-of-the-upper-bar-around-O1" data-toc-modified-id="Sum-of-moments-of-the-upper-bar-around-O1-4.9.6"><span class="toc-item-num">4.9.6&nbsp;&nbsp;</span>Sum of moments of the upper bar around O1</a></span></li><li><span><a href="#Derivative-of-angular-momentum-of-the-upper-bar-around-O1" data-toc-modified-id="Derivative-of-angular-momentum-of-the-upper-bar-around-O1-4.9.7"><span class="toc-item-num">4.9.7&nbsp;&nbsp;</span>Derivative of angular momentum of the upper bar around O1</a></span></li><li><span><a href="#The-equations-of-motion" data-toc-modified-id="The-equations-of-motion-4.9.8"><span class="toc-item-num">4.9.8&nbsp;&nbsp;</span>The equations of motion</a></span></li></ul></li></ul></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-internet" data-toc-modified-id="Video-lectures-on-the-internet-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Video lectures on the internet</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>References</a></span></li></ul></div>
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
        ## Equivalent systems


        A set of forces and moments is considered equivalent if its resultant force and sum of the moments computed relative to a given point are the same. Normally, we want to reduce all the forces and moments being applied to a body into a single force and a single moment.

        We have done this with particles for the resultant force. The resultant force is simply the sum of all the forces being applied to the body.

        <span class="notranslate">$\vec{F} = \sum\limits_{i=1}^n \vec{F_i}$</span>

        where <span class="notranslate">$\vec{F_i}$</span> is each force applied to the body.

        Similarly, the total moment applied to the body relative to a point O is:

        <span class="notranslate">$\vec{M_O} = \sum\limits_{i}\vec{r_{i/O}} \times \vec{F_i}$</span>

        where <span class="notranslate">$\vec{r_{i/O}}$</span> is the vector from the point O to the point where the force <span class="notranslate">$\vec{\bf{F_i}}$</span> is being applied.

        But where the resultant force should be applied in the body? If the resultant force were applied to any point different from the point O, it would produce an additional  moment to the body relative to point O. So, the resultant force must be applied to the point O.

        So, any set of forces can be reduced to a moment relative to a chosen point O and a resultant force applied to the point O.  

        To compute the resultant force and moment relative to another point O', the new moment is:

        <span class="notranslate">$\vec{M_{O'}} = \vec{M_O} + \vec{r_{O'/O}} \times \vec{F}$</span>

        And the resultant force is the same.

        It is worth to note that if the resultant force  <span class="notranslate">$\vec{F}$</span> is zero, than the moment is the same relative to any point.

        <figure><img src="./../images/equivalentSystem.png" width=800/></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Steps to draw a free-body diagram

        The steps to draw the free-body diagram of a body is very similar to the case of particles.

        1 - Draw separately each object considered in the problem. How you separate depends on what questions you want to answer.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        2 - Identify the forces acting on each object. If you are analyzing more than one object, remember the third Newton law (action and reaction), and identify where the reaction of a force is being applied. Whenever a movement of translation of the body is constrained, a force at the direction of the constraint must exist.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        3 - Identify the moments acting on each object. In the case you are analyzing more than one object, you must consider the  action and reaction law (third Newton-Euler law). Whenever a movement of rotation of the body is constrained, a moment at the direction of the constraint must exist.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        4 - Draw all the identified forces, representing them as vectors. The vectors should be represented with the origin in the object. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        5 - Draw all the identified moments, representing them as vectors. In planar movements, the moments we be orthogonal to the considered plane. In these cases, normally the moment vectors are represented as curved arrows.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        6 - If necessary, you should represent the reference frame (or references frames in case you use more than one reference frame) in the free-body diagram.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        7 - After this, you can solve the problem using the First and Second Newton-Euler Laws (see, e.g, [Newton-Euler Laws](newton_euler_equations.ipynb)) to find the motion of the body.

        <span class="notranslate">$\vec{F} = m\vec{a_{cm}} = m\frac{d^2\vec{r_{cm}}}{dt^2}$</span>

        <span class="notranslate">$\vec{M_O} = I_{zz}^{cm}\vec{\alpha} + m \vec{r_{cm/O}} \times \vec{a_{cm}}=I_{zz}^{cm}\vec{\frac{d^2\theta}{dt^2}} + m \vec{r_{cm/O}} \times \frac{d^2\vec{r_{cm}}}{dt^2}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Examples

        Below, we will see some examples of how to draw the free-body diagram and obtain the equation of motion.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Horizontal fixed bar 

        The first example is an example of statics. The bar has **no velocity** and **no acceleration**. We can find the **force** and **moment** the wall is applying to the bar.

        <figure><img src="../images/bar1.png" width=150 /></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Free-body diagram

        The free-body diagram of the bar is depicted below. At the point where the bar is connected to the wall, there is a force$\vec{F_1}$constraining the translation movement of the point O and a moment$\vec{M}$constraining the rotation of the bar.

        <figure><img src="../images/bar1FBD.png" width=250 /></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Sum of the forces and moments

        The resultant force being applied to the bar is:

        <span class="notranslate">$\vec{F} = -mg\hat{\bf{j}} + \vec{F_1}$</span>

        And the total moment in the z direction around the point O is:

        <span class="notranslate">$\vec{M_O} = \vec{r_{C/O}}\times-mg\hat{\bf{j}} + \vec{M}$</span>

        The vector from the point O to the point C is given by <span class="notranslate">$\vec{r_{C/O}} =\frac{l}{2}\hat{\bf{i}}$</span>. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Newton-Euler laws
        As the bar is fixed, all the accelerations are zero. So we can find the forces and the moment at the constraint.

        <span class="notranslate">$\vec{F} = \vec{0}$</span>

        <span class="notranslate">$-mg\hat{\bf{j}} + \vec{F_1} = \vec{0}$</span>

        <span class="notranslate">$\vec{F_1} = mg\hat{\bf{j}}$</span>

        <span class="notranslate">$\vec{M_O} = \vec{0}$</span>

        <span class="notranslate">$\vec{r_{C/O}}\times-mg\hat{\bf{j}} + \vec{M} = \vec{0}$</span>

        <span class="notranslate">$\frac{l}{2}\hat{\bf{i}}\times-mg\hat{\bf{j}} + \vec{M} = \vec{0}$</span>

        <span class="notranslate">$-\frac{mgl}{2}\hat{\bf{k}} + \vec{M} = \vec{0}$</span>

        <span class="notranslate">$\vec{M} = \frac{mgl}{2}\hat{\bf{k}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Rotating ball with drag force  


        A **basketball** has a mass **$\bf{m = 0.63}$kg** and radius equal$\bf{R = 12}$**cm**. A basketball player shots the ball from the free-throw line (**4.6 m from the basket**) with a **speed of 9.5 m/s**, **angle of 51 degrees with the court**, **height of 2 m** and **angular velocity of 42 rad/s**. At the ball is acting a **drag force**$\bf{b_l = 0.5}$**N.s/m** proportional to the modulus of the ball velocity in the opposite direction of the velocity and a **drag moment**$\bf{b_r = 0.001}$**N.m.s**, proportional and in the opposite direction of the angular velocity of the ball. Consider the moment of inertia of the ball as$\bf{I_{zz}^{cm} = \frac{2mR^2}{3}}$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Free-body diagram

        Below is depicted the free-body diagram of the ball.

        <figure><img src="../images/ballRotDrag.png" width=250 /></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Sum of forces and moments

        The resultant force being applied at the ball is:

        <span class="notranslate">$\vec{F} = -mg\hat{\bf{j}} - b_l\vec{v} = -mg\hat{\bf{j}} - b_l\frac{d\vec{r}}{dt}$</span>

        <span class="notranslate">$\vec{M_C} = - b_r\omega\hat{\bf{k}}=- b_r\frac{d\theta}{dt}\hat{\bf{k}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Newton-Euler laws

        So, by the second Newton-Euler law:

        <span class="notranslate">$I_{zz}^{C}\frac{d^2\theta}{dt^2}=\vec{M_C}$</span>

        <span class="notranslate">$I_{zz}^{C}\frac{d^2\theta}{dt^2} = - b_r\frac{d\theta}{dt}$</span>

        and by the first Newton-Euler law (for a revision on Newton-Euler laws, [see this notebook](newton_euler_equations.ipynb)):

        <span class="notranslate">$m\frac{d^2\vec{r}}{dt^2}=\vec{F} \rightarrow \frac{d^2\vec{r}}{dt^2} = -g\hat{\bf{j}} - \frac{b_l}{m}\frac{d^2\vec{r}}{dt^2}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Equations of motion

        So, we can split the differential equations above in three equations:

        <span class="notranslate">$\frac{d^2\theta}{dt^2} = - \frac{b_r}{I_{zz}^{C}}\frac{d\theta}{dt}$</span>

        <span class="notranslate">$\frac{d^2x}{dt^2} = - \frac{b_l}{m}\frac{dx}{dt}$</span>

        <span class="notranslate">$\frac{d^2y}{dt^2} = -g - \frac{b_l}{m}\frac{dy}{dt}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Numerical solution of the equations

        To solve these equations numerically, we can split each of these equations in first-order equations and the use a numerical method to integrate them. The first-order equations we be written in a matrix form, considering the moment of inertia of the ball as <span class="notranslate">$I_{zz}^{C}=\frac{2mR^2}{3}$:$\left[\begin{array}{c}\frac{d\omega}{dt}\\\frac{dv_x}{dt}\\\frac{dv_y}{dt}\\\frac{d\theta}{dt}\\\frac{dx}{dt}\\\frac{dy}{dt} \end{array}\right] = \left[\begin{array}{c}- \frac{3b_r}{2mR^2}\omega\\- \frac{b_l}{m}v_x\\-g - \frac{b_l}{m}v_y\\\omega\\v_x\\v_y\end{array}\right]$</span>

        Below, the equations were solved numerically by using the Euler method (for a revision on numerical methods to solve ordinary differential equations, [see this notebook](OrdinaryDifferentialEquation.ipynb)).
        """
    )
    return


@app.cell
def _(np, plt):
    m = 0.63
    R = 0.12
    I = 2.0/3*m*R**2

    bl = 0.5
    br = 0.001
    g = 9.81

    x0 = 0
    y0 = 2

    v0 = 9.5
    angle = 51*np.pi/180.0

    vx0 = v0*np.cos(angle)
    vy0 = v0*np.sin(angle)

    dt = 0.001
    t = np.arange(0, 2.1, dt)

    x = x0
    y = y0
    vx = vx0
    vy = vy0

    omega = 42
    theta = 0

    r = np.array([x,y])
    ballAngle = np.array([theta])

    state = np.array([omega, vx, vy, theta, x, y])

    while state[4]<=4.6:
        dstatedt = np.array([-br/I*state[0],-bl/m*state[1], -g-bl/m*state[2],state[0], state[1], state[2] ])
        state = state + dt * dstatedt
    
        r = np.vstack((r, [state[4], state[5]]))
        ballAngle = np.vstack((ballAngle, [state[3]]))
    
    plt.figure(figsize=(10,8))
    plt.plot(r[0:-1:50,0], r[0:-1:50,1], marker='o', color=np.array([1, 0.6,0]), markersize=17, linestyle='')
    plt.plot(np.array([4, 4.45]), np.array([3.05, 3.05]))
    for i in range(len(r[0:-1:50,0])):
        plt.plot(r[i*50,0]+np.array([-0.05*(np.cos(ballAngle[i*50])-np.sin(ballAngle[i*50])),
                                     0.05*(np.cos(ballAngle[i*50])-np.sin(ballAngle[i*50]))]), 
                 r[i*50,1] + np.array([-0.05*(np.sin(ballAngle[i*50])+np.cos(ballAngle[i*50])),
                                       0.05*(np.sin(ballAngle[i*50])+np.cos(ballAngle[i*50]))]),'k')
    plt.ylim((0,4.5))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Above is the trajectory of the ball until it reaches the basket (height of 3.05 m, marked with a blue line).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Pendulum 

        Now, we will analyze a **pendulum** of **lenght**$\bf{l}$. It consists of a bar with its **upper part linked to a hinge**.

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/pendulum.png?raw=1" width=350 alt="Pendulum"/></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Free-body diagram

        Below is the free-body diagram of the bar. On the center of mass of the bar is acting the gravitational force. On the point O, where there is a hinge, a force$\vec{\bf{F_1}}$restrains the point to translate. As the hinge does not constrains a rotational movement of the bar, there is no moment applied by the hinge. 

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/pendulumFBD.png?raw=1" width=300 alt="Pendulum FBD"/></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Sum of the forces and moments

        To find the equation of motion of the bar we can use the second Newton-Euler law. So, we must find the sum of the moments and the resultant force being applied to the bar. The sum of moments could be computed relative to any point, but if we choose the fixed point O, it is easier because we can ignore the force <span class="notranslate">$\vec{\bf{F_1}}$</span>.

        The moment around the fixed point is:

        <span class="notranslate">$\vec{M_O} = \vec{r_{cm/O}} \times (-mg\hat{\bf{j}})$</span>

        The resultant force applied in the bar is:

        <span class="notranslate">$\vec{F} = -mg\hat{\bf{j}} + \vec{F_1}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Second Newton-Euler law

        The second Newton-Euler law is:

        <span class="notranslate">$\vec{M_O} = \frac{d\vec{H_O}}{dt}$</span>
    
        The angular momentum derivative of the bar around point O is:

        <span class="notranslate">$\frac{d\vec{H_O}}{dt} = I_{zz}^{cm} \frac{d^2\theta}{dt^2} \hat{\bf{k}}  + m \vec{r_{cm/O}} \times \vec{a_{cm}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Kinematics

        The vector from point O to the center of mass is:

        <span class="notranslate">$\vec{r_{cm/O}} = \frac{l}{2}\sin{\theta}\hat{\bf{i}}-\frac{l}{2}\cos{\theta}\hat{\bf{j}}$</span>

        The position of the center of mass is, considering the point O as the origin, equal to <span class="notranslate">$\vec{\bf{r_{cm/O}}}$</span>. So, the center of mass acceleration is obtained by deriving it twice. 

        <span class="notranslate">$\vec{v_{cm}} = \frac{\vec{r_{cm/O}}}{dt} = \frac{l}{2}(\cos{\theta}\hat{\bf{i}}+\sin{\theta}\hat{\bf{j}})\frac{d\theta}{dt}$</span> 

        <span class="notranslate">$\vec{a_{cm}} = \frac{l}{2}(-\sin{\theta}\hat{\bf{i}}+\cos{\theta}\hat{\bf{j}})\left(\frac{d\theta}{dt}\right)^2 + \frac{l}{2}(\cos{\theta}\hat{\bf{i}}+\sin{\theta}\hat{\bf{j}})\frac{d^2\theta}{dt^2}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Sum of the moments

        So, the moment around the point O:$\vec{M_O} = \vec{r_{cm/O}} \times (-mg\hat{\bf{j}}) = \left(\frac{l}{2}\sin{\theta}\hat{\bf{i}}-\frac{l}{2}\cos{\theta}\hat{\bf{j}}\right) \times (-mg\hat{\bf{j}}) = \frac{-mgl}{2}\sin{\theta}\hat{\bf{k}}$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Derivative of the angular momentum

        And the derivative of the angular momentum is:

        <span class="notranslate">$\begin{array}{l l}
        \frac{d\vec{H_O}}{dt} &=& I_{zz}^{cm} \frac{d^2\theta}{dt^2}\hat{\bf{k}}  + m \frac{l}{2}(\sin{\theta}\hat{\bf{i}}-\cos{\theta}\hat{\bf{j}}) \times \left[ \frac{l}{2}(-\sin{\theta}\hat{\bf{i}}+\cos{\theta}\hat{\bf{j}})\left(\frac{d\theta}{dt}\right)^2 + \frac{l}{2}(\cos{\theta}\hat{\bf{i}}+\sin{\theta}\hat{\bf{j}})\frac{d^2\theta}{dt^2}    \right] \\
        &=& I_{zz}^{cm} \frac{d^2\theta}{dt^2}\hat{\bf{k}}  + m \frac{l^2}{4}\frac{d^2\theta}{dt^2} \hat{\bf{k}}  \\
        &=& \left(I_{zz}^{cm} + \frac{ml^2}{4}\right)\frac{d^2\theta}{dt^2} \hat{\bf{k}}
        \end{array}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### The equation of motion

        Now, by using the Newton-Euler laws, we can obtain the differential equation that describes the bar angle along time:

        <span class="notranslate">$\frac{d\vec{H_O}}{dt} = \vec{M_O} \rightarrow \frac{-mgl}{2}\sin{\theta} = \left(I_{zz}^{cm} + \frac{ml^2}{4}\right)\frac{d^2\theta}{dt^2} \rightarrow \frac{d^2\theta}{dt^2} = \frac{-2mgl}{\left(4I_{zz}^{cm} + ml^2\right)}\sin{\theta}$</span>

        The moment of inertia of the bar relative to its center of mass is <span class="notranslate">$I_{zz}^{cm} = \frac{ml^2}{12}$</span> (for a revision on moment of inertia see [this notebook](CenterOfMassAndMomentOfInertia.ipynb)).  So, the equation of motion of the pendulum is:

        <span class="notranslate">$\frac{d^2\theta}{dt^2} = \frac{-3g}{2l}\sin{\theta}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Inverted Pendulum

        At this example we analyze the inverted pendulum. It consists of a bar with **length**$\bf{l}$with its lowest extremity linked to a hinge.

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/invertedPendulum.png?raw=1" width=350 alt="Inverted Pendulum"/></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Free-body diagram

        The free-body diagram of the bar is depicted below. At the point where the bar is linked to the hinge, a force <span class="notranslate">$\vec{F_1}$</span> acts at the bar due to the restraint of the translation imposed to the point O by the hinge. Additionally, the gravitational force acts at the center of mass of the bar.
   
        <figure><img src="../images/invertedPendulumFBD.png" width=300 /></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Sum of the forces and moments

        Similarly to pendulum, in this case we can find the equation of motion of the bar by using the second Newton-Euler law. To do that we must find the sum of the moments and the resultant force being applied to the bar. The sum of moments could be computed relative to any point, but if we choose the fixed point O, it is easier because we can ignore the force <span class="notranslate">$\vec{F_1}$</span>.

        The moment around the fixed point is:

        <span class="notranslate">$\vec{M_O} = \vec{r_{cm/O}} \times (-mg\hat{\bf{j}})$</span>

        The resultant force applied in the bar is:

        <span class="notranslate">$\vec{F} = -mg\hat{\bf{j}} + \vec{F_1}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Second Newton-Euler law

        The second Newton-Euler law is:

        <span class="notranslate">$\vec{M_O} = \frac{d\vec{H_O}}{dt}$</span>

        The angular momentum derivative of the bar around point O is:

        <span class="notranslate">$\frac{d\vec{H_O}}{dt} = I_{zz}^{cm} \frac{d^2\theta}{dt^2} \hat{\bf{k}}  + m \vec{r_{cm/O}} \times \vec{a_{cm}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Kinematics

        The equations above are exactly the same of the pendulum example. The difference is in the kinematics of the bar. Now,the vector from point O to the center of mass is:

        <span class="notranslate">$\vec{r_{cm/O}} = -\frac{l}{2}\sin{\theta}\hat{\bf{i}}+\frac{l}{2}\cos{\theta}\hat{\bf{j}}$</span>

        The position of the center of mass of the bar is equal to the vector <span class="notranslate">$\vec{\bf{r_{cm/O}}}$</span>, since the point O is with velocity zero relative to the global reference frame. So the center of mass acceleration can be obtained by deriving this vector twice:

        <span class="notranslate">$\vec{v_{cm}} = \frac{\vec{r_{cm/O}}}{dt} = -\frac{l}{2}(\cos{\theta}\hat{\bf{i}}+\sin{\theta}\hat{\bf{j}})\frac{d\theta}{dt}$</span> 

        <span class="notranslate">$\vec{a_{cm}} = \frac{l}{2}(\sin{\theta}\hat{\bf{i}}-\cos{\theta}\hat{\bf{j}})\left(\frac{d\theta}{dt}\right)^2 - \frac{l}{2}(\cos{\theta}\hat{\bf{i}}+\sin{\theta}\hat{\bf{j}})\frac{d^2\theta}{dt^2}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Sum of the moments


        So, the moment around the point O is:

        <span class="notranslate">$\vec{M_O} = \vec{r_{cm/O}} \times (-mg\hat{\bf{j}}) = \left(-\frac{l}{2}\sin{\theta}\hat{\bf{i}}+\frac{l}{2}\cos{\theta}\hat{\bf{j}}\right) \times (-mg\hat{\bf{j}}) = \frac{mgl}{2}\sin{\theta} \hat{\bf{k}}$</span>
    
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        #### Derivative of the angular momentum


        And the derivative of the angular momentum is:

        <span class="notranslate">$\begin{array}{l l}
        \frac{d\vec{H_O}}{dt} &=& I_{zz}^{cm} \frac{d^2\theta}{dt^2} \hat{\bf{k}} + m \left(-\frac{l}{2}\sin{\theta}\hat{\bf{i}}+\frac{l}{2}\cos{\theta}\hat{\bf{j}}\right) \times \left[\frac{l}{2}(\sin{\theta}\hat{\bf{i}}-\cos{\theta}\hat{\bf{j}})\left(\frac{d\theta}{dt}\right)^2 - \frac{l}{2}(\cos{\theta}\hat{\bf{i}}+\sin{\theta}\hat{\bf{j}})\frac{d^2\theta}{dt^2}\right] \\
        &=& I_{zz}^{cm} \frac{d^2\theta}{dt^2} \hat{\bf{k}} + m \frac{l^2}{4}\frac{d^2\theta}{dt^2}\hat{\bf{k}} \\
        &=& \left(I_{zz}^{cm} + \frac{ml^2}{4}\right)\frac{d^2\theta}{dt^2}\hat{\bf{k}}
        \end{array}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### The equation of motion


        By using the Newton-Euler laws, we can find the equation of motion of the bar:

        <span class="notranslate">$\frac{d\vec{H_O}}{dt} = \vec{M_O} \rightarrow \left(I_{zz}^{cm} +  \frac{ml^2}{4}\right)\frac{d^2\theta}{dt^2} = \frac{mgl}{2}\sin{\theta} \rightarrow \frac{d^2\theta}{dt^2} = \frac{2mgl}{\left(4I_{zz}^{cm} + ml^2\right)}\sin(\theta)$</span>

        The moment of inertia of the bar is <span class="notranslate">$I_{zz}^{cm} = \frac{ml^2}{12}$</span>. So, the equation of motion of the bar is:

        <span class="notranslate">$\frac{d^2\theta}{dt^2} = \frac{3g}{2l}\sin(\theta)$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Human quiet standing 

        A very simple model of the human quiet standing (frequently used) is to use an inverted pendulum to model the human body. On [this notebook](http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/IP_Model.ipynb) there is a more comprehensive explanation of human standing. Consider the body as a rigid uniform bar with moment of inertia <span class="notranslate">$I_{zz}^{cm} = \frac{m_Bh_G^2}{3}$</span>. 

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/quietStanding.png?raw=1" width=300 alt="Quiet Standing"/></figure>

        Adapted from [Elias, Watanabe and Kohn (2014)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003944).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Free-body diagram

        Below is depicted the free-body diagram of the foot and the rest of the body. At the ankle joint there is a constraint to the translation of the ankle joint. So, a force$\vec{\bf{F_1}}$is applied to the body at the ankle joint. By the third Newton law, a force$-\vec{\bf{F_1}}$is applied to the foot at the ankle joint. At the center of mass of the body and of the foot, gravitational forces are applied. Additionally, a ground reaction force is applied to the foot at the center of pressure of the forces applied at the foot.

        Additionally, a moment <span class="notranslate">$\vec{T_A}$</span> is applied at the body and its reaction is applied to the foot. This moment <span class="notranslate">$\vec{T_A}$</span> comes from the muscles around the ankle joint. It is usual in Biomechanics to represent the net torque generated by all the muscles on a single joint as a single moment applied to the body. 
    
        <figure><img src="../images/quietStandingFBD.png" width=300 /></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Sum of the moments

        The process to obtain the equation of motion of the body is very similar to the pendulums. The moment around the ankle being applied to the body is:

        <span class="notranslate">$\vec{M_A} = \vec{T_A} + \vec{r_{cm/A}} \times (-m_Bg\hat{\bf{j}})$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Second Newton-Euler law

        And the derivative of the angular momentum is:

        <span class="notranslate">$\frac{d\vec{H_A}}{dt} = I_{zz}^{cm}\frac{d^2\theta_A}{dt^2} + m\vec{r_{cm/A}} \times \vec{a_{cm}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Kinematics

        To find the kinematics of the bar, we could do the same procedure we have used in the pendulum and inverted pendulum examples, but this time we will use polar coordinates (for a revision on polar coordinates, see [Polar coordinates notebook](PolarCoordinates.ipynb)). 

        <span class="notranslate">$\vec{r_{cm/A}} = \vec{r_{cm}} = h_G\hat{\bf{e_r}}$</span> 

        <span class="notranslate">$\vec{v_{cm}} = h_G\frac{d\theta_A}{dt}\hat{\bf{e_\theta}}$</span> 

        <span class="notranslate">$\vec{a_{cm}} = -h_G\left(\frac{d\theta_A}{dt}\right)^2\hat{\bf{e_r}} + h_G\frac{d^2\theta_A}{dt^2}\hat{\bf{e_\theta}}$</span>

        where <span class="notranslate">$\hat{\bf{e_r}} = -\sin(\theta_A)\hat{\bf{i}} + \cos(\theta_A)\hat{\bf{j}}$</span> and <span class="notranslate">$\hat{\bf{e_\theta}} = -\cos(\theta_A)\hat{\bf{i}} - \sin(\theta_A)\hat{\bf{j}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Sum of the moments

        Having the kinematics computed, we can go back to the moment and derivative of the angular momentum:

        <span class="notranslate">$\vec{M_A} =  \vec{T_A} + h_G\hat{\bf{e_r}} \times (-m_Bg\hat{\bf{j}}) = T_A\hat{\bf{k}} + h_Gm_Bg\sin(\theta_A)\hat{\bf{k}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Derivative of the angular momentum

        <span class="notranslate">$\begin{array}{l l}
        \frac{d\vec{H_A}}{dt} &=& I_{zz}^{cm}\frac{d^2\theta_A}{dt^2} \hat{\bf{k}} + mh_G\hat{\bf{e_r}} \times \left(-h_G\left(\frac{d^2\theta_A}{dt^2}\right)^2\hat{\bf{e_r}} + h_G\frac{d^2\theta_A}{dt^2}\hat{\bf{e_\theta}}\right) \\
        &=& I_{zz}^{cm}\frac{d^2\theta_A}{dt^2}\hat{\bf{k}} + mh_G^2\frac{d^2\theta_A}{dt^2}\hat{\bf{k}} \\
        &=& \left(I_{zz}^{cm} + mh_G^2\right)\frac{d^2\theta_A}{dt^2}\hat{\bf{k}}
        \end{array}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### The equation of motion

        By using the Newton-Euler equations, we can now find the equation of motion of the body during quiet standing:

        <span class="notranslate">$\vec{M_A}=\frac{d\vec{H_A}}{dt}$</span> 

        <span class="notranslate">$\left(I_{zz}^{cm} + mh_G^2\right)\frac{d^2\theta_A}{dt^2} =  T_A + h_Gm_B g\sin(\theta_A)$</span> 

        <span class="notranslate">$\frac{d^2\theta_A}{dt^2} = \frac{h_Gm_B g}{I_{zz}^{cm} + mh_G^2}\sin(\theta_A)+ \frac{T_A}{I_{zz}^{cm} + m_Bh_G^2}$</span>

        If we consider the body as a rigid uniform bar, the moment of inertia is <span class="notranslate">$I_{zz}^{cm} = \frac{m_Bh_G^2}{3}$</span>. So, the equation of motion of the body is:

        <span class="notranslate">$\frac{d^2\theta_A}{dt^2} = \frac{3 g}{4h_g}\sin(\theta_A)+ \frac{3 T_A}{4m_bh_G^2}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Force platform (position of the center of pressure)

        From the same example above, now we will find the position of the center of pressure ($COP_x$) during the quiet standing. The$COP_x$is the point where the ground reaction force is being applied. The point O is linked to the ground in a way that it constraints the translation and rotation movement of the platform and is located at known distance$y_0$below the ground. Also, it is in the point O that sensors of force and moments are located. 

        <br>
        <figure><img src="../images/forcePlatform2DS.png" width=250 /></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Free-body diagram

        The free-body diagram of the foot and the force platform is depicted below.

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/forcePlatform2D.png?raw=1" width=300 alt="Force Platform 2D"/></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the platform , the distance <span class="notranslate">$y_0$</span> is known. As the platform is in equilibrium its derivative of the angular momentum is zero. Now, we must find the sum of the moments. As we can choose any point, we will choose, the point where the force <span class="notranslate">$\vec{\bf{F}}$</span> is being applied, and equal it to zero.

        <span class="notranslate">$\vec{M} + \vec{r_{COP/O}} \times (-\vec{GRF}) = 0$</span>

        The vector <span class="notranslate">$\vec{\bf{r_{COP/O}}}$</span> is:

        <span class="notranslate">$\vec{r_{COP/O}} = COP_x \hat{\bf{i}} + y_0 \hat{\bf{j}}$</span>

        So, from the equation of the sum of the moments we can isolate the postion of the center of pressure:

        <span class="notranslate">$M - COP_x GRF_y + y_0 GRF_x = 0 \rightarrow COP_x = \frac{M+y_0 GRF_x}{GRF_y}$</span>
    
        Using the expression above, we can track the position of the center of pressure during an experiment.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Standing dummy (from Ruina and Pratap book)

        A standing dummy is modeled as having massless rigid circular feet of radius$R$rigidly attached to their
        uniform rigid body of length <span class="notranslate">$L$</span> and mass <span class="notranslate">$m$</span>. The feet do not slip on the floor.

        <figure><img src="../images/rocker.png" width=450 /></figure>
  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
  
        #### Free-body diagram

        Below is depicted the fee-body diagram. The point C is the point of contact with the ground. Note that this point changes its position along time, but instantly it has velocity zero.

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/rockerFBD.png?raw=1" width=450 alt="Rocker FBD"/></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Kinematics

        Using the versors:

        <span class="notranslate">$\hat{\bf{e_r}} = -\sin(\theta_A)\hat{\bf{i}} + \cos(\theta_A)\hat{\bf{j}}$</span> 

        <span class="notranslate">$\hat{\bf{e_\theta}} = -\cos(\theta_A)\hat{\bf{i}} - \sin(\theta_A)\hat{\bf{j}}$</span>

        The vector from the point C to the center of mass is:

        <span class="notranslate">$\vec{r_{cm/C}} = R\hat{\bf{j}} + \left(\frac{L}{2} - R \right)\hat{\bf{e_r}}$</span>

        The position of the center of mass, relative to the origin is different from$\vec{r_{cm/C}}$. We can consider the origin of the system as the point of contact with the ground when the standing dummy is at the vertical position.

        <span class="notranslate">$\vec{r_{cm}} = -R\theta\hat{\bf{i}} + R\hat{\bf{j}} + \left(\frac{L}{2} - R \right)\hat{\bf{e_r}}$</span> 

        <span class="notranslate">$\vec{v_{cm}} = -R\dot{\theta}\hat{\bf{i}} + \left(\frac{L}{2} - R \right)\dot{\theta}\hat{\bf{e_\theta}}$</span> 

        <span class="notranslate">$\vec{a_{cm}} = -R\ddot{\theta}\hat{\bf{i}} + \left(\frac{L}{2} - R \right)\ddot{\theta}\hat{\bf{e_\theta}} - \left(\frac{L}{2} - R \right)\dot{\theta}^2\hat{\bf{e_r}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Sum of the moments


        So, the moment around the point C is:

        <span class="notranslate">$\vec{M_C} = \vec{r_{cm/C}} \times (-mg\hat{\bf{j}}) = \left[R\hat{\bf{j}} + \left(\frac{L}{2} - R \right)\hat{\bf{e_r}} \right] \times (-mg\hat{\bf{j}})$</span> 

        <span class="notranslate">$\vec{M_C} = mg\left(\frac{L}{2} - R \right)\sin{\theta}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Derivative of the angular momentum

        <span class="notranslate">$\begin{array}{l l}
        \dfrac{d\vec{H_C}}{dt} &=& I_{zz}^{cm}\ddot{\theta} \hat{\bf{k}} + m\vec{\bf{r_{cm/C}}} \times \vec{\bf{a_{cm}}} \\
        &=& \left[mR\hat{\bf{j}} + m\left(\frac{L}{2} - R \right)\hat{\bf{e_r}}\right] \times \left[-R\ddot{\theta}\hat{\bf{i}} + \left(\frac{L}{2} - R \right)\ddot{\theta}\hat{\bf{e_\theta}} - \left(\frac{L}{2} - R \right)\dot{\theta}^2\hat{\bf{e_r}}\right]
        \end{array}$</span> 

        <span class="notranslate">$\frac{d\vec{H_C}}{dt} = \left(I_{zz}^{cm}\ddot{\theta} + mR^2\ddot{\theta} + mR\left(\frac{L}{2} - mR \right)\ddot{\theta}\cos{\theta} - mR\left(\frac{L}{2} - mR \right)\dot{\theta}^2\sin{\theta} + mR\ddot{\theta}\left(\frac{L}{2} - R \right)\cos{\theta}+m\left(\frac{L}{2} - R \right)^2\ddot{\theta}\right)\hat{\bf{k}} = \left(I_{zz}^{cm}\ddot{\theta}+mR^2\ddot{\theta} + 2mR\left(\frac{L}{2} - R \right)\ddot{\theta}\cos{\theta} - mR\left(\frac{L}{2} - R \right)\dot{\theta}^2\sin{\theta} + m\left(\frac{L}{2} - R \right)^2\ddot{\theta}\right)\hat{\bf{k}} = \left(I_{zz}^{cm}\ddot{\theta} + mR^2\ddot{\theta} + m\left(RL - 2R^2 \right)\ddot{\theta}\cos{\theta} - m\left(\frac{RL}{2} - R^2 \right)\dot{\theta}^2\sin{\theta} + m\left(\frac{L^2}{4} - LR + R^2 \right)\ddot{\theta}\right)\hat{\bf{k}}$</span> 

        <span class="notranslate">$\frac{d\vec{H_C}}{dt} = \left(I_{zz}^{cm}+2mR^2 + mRL\cos{\theta} - 2mR^2\cos{\theta} + m\frac{L^2}{4}-mLR\right)\ddot{\theta} - m\left(\frac{RL}{2} - R^2 \right)\dot{\theta}^2\sin{\theta}$</span>

        Considering the moment of inertia of the bar as <span class="notranslate">$I_{zz}^{cm} = \frac{mL^2}{12}$</span>:

        <span class="notranslate">$\frac{d\vec{H_C}}{dt} = \left(\frac{mL^2}{12}+2mR^2 + mRL\cos{\theta} - 2mR^2\cos{\theta} + \frac{mL^2}{4}-mLR\right)\ddot{\theta} - m\left(\frac{RL}{2} - R^2 \right)\dot{\theta}^2\sin{\theta}$</span> 

        <span class="notranslate">$\frac{d\vec{H_C}}{dt} = \left(\frac{mL^2}{3}+(2mR^2- mRL)(1-\cos{\theta})\right)\ddot{\theta} - m\left(\frac{RL}{2} - R^2 \right)\dot{\theta}^2\sin{\theta}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### The equation of motion

        By the second Newton-Euler law:

        <span class="notranslate">$\left(2mL^2+(12mR^2- 6mRL)(1-\cos{\theta})\right)\ddot{\theta} - m\left(3RL - 6R^2 \right)\dot{\theta}^2\sin{\theta} = mg\left(3L - 6R \right)\sin{\theta}$</span> 

        <span class="notranslate">$\left(2L^2+(12R^2- 6RL)(1-\cos{\theta})\right)\ddot{\theta} = g\left(3L - 6R \right)\sin{\theta} + \left(3RL - 6R^2 \right)\dot{\theta}^2\sin{\theta}$</span> 

        <span class="notranslate">$\ddot{\theta} = \frac{3\left(L - 2R \right)\sin{\theta}\left(g + R\dot{\theta}^2\right)}{2L^2+(12R^2- 6RL)(1-\cos{\theta})}$</span>

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Double pendulum with actuators

        The figure below shows a  double pendulum. This model can represent, for example, the arm and forearm of a subject.

        <figure><img src="../images/doublePendulum.png" width=350 /></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Free-body diagram

        The free-body diagrams of both bars are depicted below. There are constraint  forces at all joints and moments from the muscles from both joints. 
    
        <figure><img src="../images/doublePendulumFBD.png" width=700 /></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Definition of versors

        The versors <span class="notranslate">$\hat{\bf{e_{R_1}}}$,$\hat{\bf{e_{R_2}}}$,$\hat{\bf{e_{\theta_1}}}$</span> and <span class="notranslate">$\hat{\bf{e_{\theta_2}}}$</span> are defined as:

        <span class="notranslate">$\hat{\bf{e_{R_1}}} = \sin(\theta_1) \hat{\bf{i}} - \cos(\theta_1) \hat{\bf{j}}$</span> 

        <span class="notranslate">$\hat{\bf{e_{\theta_1}}} = \cos(\theta_1) \hat{\bf{i}} + \sin(\theta_1) \hat{\bf{j}}$</span> 

        <span class="notranslate">$\hat{\bf{e_{R_2}}} = \sin(\theta_2) \hat{\bf{i}} - \cos(\theta_2) \hat{\bf{j}}$</span> 

        <span class="notranslate">$\hat{\bf{e_{\theta_2}}} = \cos(\theta_2) \hat{\bf{i}} + \sin(\theta_2) \hat{\bf{j}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Kinematics

        <span class="notranslate">$\vec{r_{C1/O1}} = \frac{l_1}{2}  \hat{\bf{e_{R_1}}}$</span> 

        <span class="notranslate">$\vec{r_{O2/O1}} = l_1\hat{\bf{e_{R_1}}}$</span> 

        <span class="notranslate">$\vec{r_{C1}} = \vec{r_{C1/O1}} = \frac{l_1}{2}  \hat{\bf{e_{R_1}}}$</span> 

        <span class="notranslate">$\vec{v_{C1}} = \frac{l_1}{2}\frac{d\theta_1}{dt}\hat{\bf{e_{\theta_1}}}$</span> 

        <span class="notranslate">$\vec{a_{C1}} = -\frac{l_1}{2}\left(\frac{d\theta_1}{dt}\right)^2 \hat{\bf{e_{R_1}}} + \frac{l_1}{2}\frac{d^2\theta_1}{dt^2} \hat{\bf{e_{\theta_1}}}$</span> 

        <span class="notranslate">$\vec{r_{C2/O2}} = \frac{l_2}{2}\hat{\bf{e_{R_2}}}$</span> 

        <span class="notranslate">$\vec{r_{C2}} = \vec{r_{O2/O1}} + \vec{r_{C2/O2}} =  l_1\hat{\bf{e_{R_1}}} + \frac{l_2}{2}\hat{\bf{e_{R_2}}}$</span> 

        <span class="notranslate">$\vec{v_{C2}} = l_1\frac{d\theta_1}{dt}\hat{\bf{e_{\theta_1}}} + \frac{l_2}{2}\frac{d\theta_1}{dt}\hat{\bf{e_{\theta_2}}}$</span> 

        <span class="notranslate">$\vec{a_{C2}} = -l_1\left(\frac{d\theta_1}{dt}\right)^2 \hat{\bf{e_{R_1}}} + l_1\frac{d^2\theta_1}{dt^2} \hat{\bf{e_{\theta_1}}} -\frac{l_2}{2}\left(\frac{d\theta_2}{dt}\right)^2 \hat{\bf{e_{R_2}}} + \frac{l_2}{2}\frac{d^2\theta_2}{dt^2} \hat{\bf{e_{\theta_2}}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Sum of the moments of the lower bar around O2

        First, we can analyze the the sum of the moments and the derivative of the angular momentum at the lower bar relative to the point O2.

        <span class="notranslate">$\vec{M_{O2}} = \vec{r_{C2/O2}} \times (-m_2g\hat{\bf{j}}) + \vec{M_{12}} =  \frac{l_2}{2}   \hat{\bf{e_{R_2}}} \times (-m_2g\hat{\bf{j}}) + \vec{M_{12}} = -\frac{m_2gl_2}{2}\sin(\theta_2)\hat{\bf{k}} + \vec{M_{12}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Derivative of angular momentum of the lower bar around O2

        <span class="notranslate">$\frac{d\vec{H_{O2}}}{dt} = I_{zz}^{C2} \frac{d^2\theta_2}{dt^2} + m_2 \vec{r_{C2/O2}} \times \vec{a_{C2}} = I_{zz}^{C2} \frac{d^2\theta_2}{dt^2} + m_2 \frac{l_2}{2}  \hat{\bf{e_{R_2}}} \times \left[-l_1\left(\frac{d\theta_1}{dt}\right)^2 \hat{\bf{e_{R_1}}} + l_1\frac{d^2\theta_1}{dt^2} \hat{\bf{e_{\theta_1}}} -\frac{l_2}{2}\left(\frac{d\theta_2}{dt}\right)^2 \hat{\bf{e_{R_2}}} + \frac{l_2}{2}\frac{d^2\theta_2}{dt^2} \hat{\bf{e_{\theta_2}}}\right]$</span>

        To complete the computation of the derivative of the angular momentum we must find the products <span class="notranslate">$\hat{\bf{e_{R_2}}} \times \hat{\bf{e_{R_1}}}$,$\hat{\bf{e_{R_2}}} \times \hat{\bf{e_{\theta_1}}}$</span> and <span class="notranslate">$\hat{\bf{e_{R_1}}} \times \hat{\bf{e_{\theta_2}}}$</span>:

        <span class="notranslate">$\hat{\bf{e_{R_2}}} \times \hat{\bf{e_{\theta_1}}} = \left[\begin{array}{c}\sin(\theta_2)\\-\cos(\theta_2)\end{array}\right] \times \left[\begin{array}{c}\cos(\theta_1)\\\sin(\theta_1)\end{array}\right] = \sin(\theta_1)\sin(\theta_2)+\cos(\theta_1)\cos(\theta_2) = \cos(\theta_1-\theta_2)$</span>

        <span class="notranslate">$\hat{\bf{e_{R_2}}} \times \hat{\bf{e_{R_1}}} = \left[\begin{array}{c}\sin(\theta_2)\\-\cos(\theta_2)\end{array}\right] \times \left[\begin{array}{c}\sin(\theta_1)\\-\cos(\theta_1)\end{array}\right] = -\sin(\theta_2)\cos(\theta_1)+\cos(\theta_2)\sin(\theta_1) = \sin(\theta_1-\theta_2)$</span>

        <span class="notranslate">$\hat{\bf{e_{R_1}}} \times \hat{\bf{e_{\theta_2}}} =\left[\begin{array}{c}\sin(\theta_1)\\-\cos(\theta_1)\end{array}\right] \times \left[\begin{array}{c}\cos(\theta_2)\\\sin(\theta_2)\end{array}\right] = \sin(\theta_2)\sin(\theta_1)+\cos(\theta_2)\cos(\theta_1) = \cos(\theta_1-\theta_2)$</span>
    
        So, the derivative of the angular momentum is:

        <span class="notranslate">
        \begin{align}
        \begin{split}
            \frac{d\vec{H_{O2}}}{dt} &=  I_{zz}^{C2} \frac{d^2\theta_2}{dt^2} - \frac{m_2l_1l_2}{2}\left(\frac{d\theta_1}{dt}\right)^2 \sin(\theta_1-\theta_2) + \frac{m_2l_1l_2}{2}\frac{d^2\theta_1}{dt^2} \cos(\theta_1-\theta_2) + \frac{m_2l_2^2}{4}\frac{d^2\theta_2}{dt^2} \\
            &=  \frac{m_2l_1l_2}{2}\cos(\theta_1-\theta_2)\frac{d^2\theta_1}{dt^2} + \left(I_{zz}^{C2} + \frac{m_2l_2^2}{4} \right)\frac{d^2\theta_2}{dt^2}- \frac{m_2l_1l_2}{2}\left(\frac{d\theta_1}{dt}\right)^2 \sin(\theta_1-\theta_2) 
            \end{split}
        \end{align}
        </span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Sum of moments of the upper bar around O1

        Now we can analyze the sum of the moments and the derivative of the angular momentum at the upper bar. They will be computed relative to the point O1. 

        <span class="notranslate">$\begin{array}{l l}
        \vec{M_{O1}} &=& \vec{r_{C1/O1}} \times (-m_1g\hat{\bf{j}}) + \vec{r_{O2/O1}} \times (-\vec{F_{12}}) + \vec{M_{1}} - \vec{M_{12}} \\
        &=& \frac{l_1}{2}  \hat{\bf{e_{R_1}}}  \times (-m_1g\hat{\bf{j}}) + l_1\hat{\bf{e_{R_1}}} \times (-\vec{F_{12}}) + \vec{M_{1}} - \vec{M_{12}}
        \end{array}$</span>

        <span class="notranslate">$\vec{M_{O1}}= \frac{-m_1gl_1}{2}\sin(\theta_1)\hat{\bf{k}} + l_1\hat{\bf{e_{R_1}}} \times (-\vec{F_{12}}) + \vec{M_{1}} - \vec{M_{12}}$</span>

        It remains to find the force <span class="notranslate">$\vec{F_{12}}$</span> that is in the sum of moment of the upper bar. It can be found by finding the resultant force in the lower bar and use the First Newton-Euler law:

        <span class="notranslate">$\vec{F_{12}} - m_2g \hat{\bf{j}} = m\vec{a_{C2}}$</span>

        <span class="notranslate">$\vec{F_{12}}  =  m_2g \hat{\bf{j}}  + m_2\left[-l_1\left(\frac{d\theta_1}{dt}\right)^2 \hat{\bf{e_{R_1}}} + l_1\frac{d^2\theta_1}{dt^2} \hat{\bf{e_{\theta_1}}} -\frac{l_2}{2}\left(\frac{d\theta_2}{dt}\right)^2 \hat{\bf{e_{R_2}}} + \frac{l_2}{2}\frac{d^2\theta_2}{dt^2} \hat{\bf{e_{\theta_2}}}\right]$</span>

        Now, we can go back to the moment <span class="notranslate">$\vec{M_{O1}}$</span>:

        <span class="notranslate">
        \begin{align}
            \begin{split}
            \vec{M_{O1}} &= \frac{-m_1gl_1}{2}\sin(\theta_1)\hat{\bf{k}} + l_1\hat{\bf{e_{R_1}}} \times (-\vec{\bf{F_{12}}}) + \vec{M_{1}} - \vec{M_{12}} \\
            &= -\frac{m_1gl_1}{2}\sin(\theta_1)\hat{\bf{k}} - l_1\hat{\bf{e_{R_1}}} \times   m_2\left[g \hat{\bf{j}}-l_1\left(\frac{d\theta_1}{dt}\right)^2 \hat{\bf{e_{R_1}}} + l_1\frac{d^2\theta_1}{dt^2} \hat{\bf{e_{\theta_1}}} -\frac{l_2}{2}\left(\frac{d\theta_2}{dt}\right)^2 \hat{\bf{e_{R_2}}} + \frac{l_2}{2}\frac{d^2\theta_2}{dt^2} \hat{\bf{e_{\theta_2}}}\right]  + \vec{M_{1}} - \vec{M_{12}} \\
            &= -\frac{m_1gl_1}{2}\sin(\theta_1)\hat{\bf{k}} - m_2l_1g \sin(\theta_1)\hat{\bf{k}}  - m_2l_1^2\frac{d^2\theta_1}{dt^2}\hat{\bf{k}}  + \frac{m_2l_1l_2}{2}\left(\frac{d\theta_2}{dt}\right)^2 \sin(\theta_1-\theta_2)\hat{\bf{k}}  - \frac{m_2l_1l_2}{2}\frac{d^2\theta_2}{dt^2} \cos(\theta_1-\theta_2)\hat{\bf{k}}  + M_1\hat{\bf{k}}  - M_{12}\hat{\bf{k}} 
            \end{split}
        \end{align}
        </span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Derivative of angular momentum of the upper bar around O1

        <span class="notranslate">$\begin{array}{l l}
        \frac{d\vec{H_{O1}}}{dt} &=& I_{zz}^{C1} \frac{d^2\theta_1}{dt^2} + m_1 \vec{r_{C1/O1}} \times \vec{a_{C1}} \\
        &=& I_{zz}^{C1} \frac{d^2\theta_1}{dt^2} + m_1 \frac{l_1}{2} \hat{\bf{e_{R_1}}} \times \left[-\frac{l_1}{2}\left(\frac{d\theta_1}{dt}\right)^2 \hat{\bf{e_{R_1}}} + \frac{l_1}{2}\frac{d^2\theta_1}{dt^2} \hat{\bf{e_{\theta_1}}}\right]
        \end{array}$</span>

        <span class="notranslate">$\frac{d\vec{H_{O1}}}{dt} =\left( I_{zz}^{C1}  + m_1 \frac{l_1^2}{4}\right)\frac{d^2\theta_1}{dt^2}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### The equations of motion

        Finally, using the second Newton-Euler law for both bars, we can find the equation of motion of both angles:

        <span class="notranslate">$\begin{array}{l l}
        \dfrac{d\vec{H_{O1}}}{dt} &=& \vec{M_{O1}} \rightarrow  \left( I_{zz}^{C1} + m_1 \frac{l_1^2}{4}\right)\frac{d^2\theta_1}{dt^2} \\
        &=& -\frac{m_1gl_1}{2}\sin(\theta_1)- m_2l_1g \sin(\theta_1) - m_2l_1^2\frac{d^2\theta_1}{dt^2}  + \frac{m_2l_1l_2}{2}\left(\frac{d\theta_2}{dt}\right)^2 \sin(\theta_1-\theta_2) - \frac{m_2l_1l_2}{2}\frac{d^2\theta_2}{dt^2} \cos{(\theta_1-\theta_2)} + M_1 - M_{12} 
        \end{array}$</span>

        <span class="notranslate">$\begin{array}{l l}
        \dfrac{d\vec{H_{O2}}}{dt} &=& \vec{M_{O2}} \rightarrow \frac{m_2l_1l_2}{2}\cos(\theta_1-\theta_2)\frac{d^2\theta_1}{dt^2} + \left(I_{zz}^{C2} + \frac{m_2l_2^2}{4} \right)\frac{d^2\theta_2}{dt^2}- \frac{m_2l_1l_2}{2}\left(\frac{d\theta_1}{dt}\right)^2 \sin(\theta_1-\theta_2) \\
        &=& -\frac{m_2gl_2}{2}\sin(\theta_2) + M_{12}
        \end{array}$</span>

        Considering the moment of inertia of the bars as$I_{zz}^{C1}=\frac{m_1l_1^2}{12}$and$I_{zz}^{C2}=\frac{m_2l_2^2}{12}$the differential equations above are:

        <span class="notranslate">$\left(\dfrac{m_1l_1^2}{3} +m_2l_1^2\right)\frac{d^2\theta_1}{dt^2} + \frac{m_2l_1l_2}{2} \cos{(\theta_1-\theta_2)\frac{d^2\theta_2}{dt^2}}  =  -\frac{m_1gl_1}{2}\sin(\theta_1)- m_2l_1g \sin(\theta_1)   + \frac{m_2l_1l_2}{2}\left(\frac{d\theta_2}{dt}\right)^2 \sin(\theta_1-\theta_2)  + M_1 - M_{12}$</span>
    
        <span class="notranslate">$\dfrac{m_2l_1l_2}{2}\cos(\theta_1-\theta_2)\frac{d^2\theta_1}{dt^2} + \frac{m_2l_2^2}{3}\frac{d^2\theta_2}{dt^2} = -\frac{m_2gl_2}{2}\sin(\theta_2) + \frac{m_2l_1l_2}{2}\left(\frac{d\theta_1}{dt}\right)^2 \sin(\theta_1-\theta_2)+ M_{12}$</span>
    
        We could isolate the angular accelerations but this would lead to very long equations.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

        - Read the chapters 17 and 19 of the [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the internet

        - [Free Body Diagram for Rigid Body Equilibrium (I)](https://www.youtube.com/watch?v=go4vsgnb9jU);
        - [Free Body Diagram for Rigid Body Equilibrium (II)](https://www.youtube.com/watch?v=Qv2WoqJlIKQ);
        - [Free Body Diagram for Rigid Body Equilibrium (III)](https://www.youtube.com/watch?v=Knk068P-COg).  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. Solve the problems 6, 7 and 8 from [this Notebook](FreeBodyDiagram.ipynb).
        2. Solve the problems 17.2.9, 19.3.26, 19.3.29 from Ruina and Pratap book.  
        3. Study the content and solve the exercises of the text [Forces and Torques in Muscles and Joints](http://cnx.org/contents/d703853c-6382-4035-8d6a-dcbca00a15ca/Forces_and_Torques_in_Muscles_) (or [download here an off-line HTML copy of the content of that website](https://github.com/BMClab/BMC/blob/master/refs/forces-and-torques-in-muscles-and-joints-14.zip?raw=true)).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Ruina A., Rudra P. (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press. 

        - Duarte M. (2017) [Free body diagram](FreeBodyDiagram.ipynb).

        - Elias, L A, Watanabe, R N, Kohn, A F.(2014) [Spinal Mechanisms May Provide a Combination of Intermittent and Continuous Control of Human Posture: Predictions from a Biologically Based Neuromusculoskeletal Model.](http://dx.doi.org/10.1371/journal.pcbi.1003944) PLOS Computational Biology (Online).

        - Winter D. A., (2009) Biomechanics and motor control of human movement. John Wiley and Sons.

        - Duarte M. (2017) [The inverted pendulum model of the human standing](http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/IP_Model.ipynb).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
