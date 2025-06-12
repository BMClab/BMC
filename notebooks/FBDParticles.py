import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Free-Body Diagram for particles

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
        <div class="toc"><ul class="toc-item"><li><span><a href="#Python-setup" data-toc-modified-id="Python-setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Python setup</a></span></li><li><span><a href="#Free-Body-Diagram" data-toc-modified-id="Free-Body-Diagram-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Free-Body Diagram</a></span></li><li><span><a href="#Steps-to-draw-a-free-body-diagram-(FBD)" data-toc-modified-id="Steps-to-draw-a-free-body-diagram-(FBD)-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Steps to draw a free-body diagram (FBD)</a></span></li><li><span><a href="#Basic-element-and-forces" data-toc-modified-id="Basic-element-and-forces-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Basic element and forces</a></span><ul class="toc-item"><li><span><a href="#Gravity" data-toc-modified-id="Gravity-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Gravity</a></span><li><span><a href="#Spring" data-toc-modified-id="Spring-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Spring</a></span><li><span><a href="#Damping" data-toc-modified-id="Damping-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Damping</a></span></li></ul><li><span><a href="#Examples-of-free-body-diagram" data-toc-modified-id="Examples-of-free-body-diagram-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Examples of free-body diagram</a></span><ul class="toc-item"><li><span><a href="#No-force-acting-on-the-particle" data-toc-modified-id="No-force-acting-on-the-particle-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>No force acting on the particle</a></span><li><span><a href="#Gravity-force-acting-on-the-particle" data-toc-modified-id="Gravity-force-acting-on-the-particle-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Gravity force acting on the particle</a><li><span><a href="#Ground-reaction-force" data-toc-modified-id="Ground-reaction-force-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Ground reaction force</a><li><span><a href="#Mass-spring-system-with-horizontal-movement" data-toc-modified-id="Mass-spring-system-with-horizontal-movement-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Mass-spring system with horizontal movement</a></span><li><span><a href="#Linear-spring-in-bidimensional-movement-at-horizontal-plane" data-toc-modified-id="Linear-spring-in-bidimensional-movement-at-horizontal-plane-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>Linear spring in bidimensional movement at horizontal plane</a></span><li><span><a href="#Particle-under-action-of-gravity-and-linear-air-resistance" data-toc-modified-id="Particle-under-action-of-gravity-and-linear-air-resistance-5.6"><span class="toc-item-num">5.6&nbsp;&nbsp;</span>Particle under action of gravity and linear air resistance</a></span><li><span><a href="#Particle-under-action-of-gravity-and-nonlinear-air-resistance" data-toc-modified-id="Particle-under-action-of-gravity-and-nonlinear-air-resistance-5.7"><span class="toc-item-num">5.7&nbsp;&nbsp;</span>Particle under action of gravity and nonlinear air resistance</a></span><li><span><a href="#Linear-spring-and-damping-on-bidimensional-horizontal-movement" data-toc-modified-id="Linear-spring-and-damping-on-bidimensional-horizontal-movement-5.8"><span class="toc-item-num">5.8&nbsp;&nbsp;</span>Linear spring and damping on bidimensional horizontal movement</a></span><li><span><a href="#Simple-muscle-model" data-toc-modified-id="Simple-muscle-model-5.9"><span class="toc-item-num">5.9&nbsp;&nbsp;</span>Simple muscle model</a></span></li></ul></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-internet" data-toc-modified-id="Video-lectures-on-the-internet-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Video lectures on the internet</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>
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
    import seaborn as sns
    # '%matplotlib inline' command supported automatically in marimo
    sns.set_context('notebook', font_scale=1.2)
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Free-Body Diagram

        In the mechanical modeling of an inanimate or living system, composed by one or more bodies (bodies as units that are mechanically isolated according to the question one is trying to answer), it is convenient to isolate each body (be they originally interconnected or not) and identify each force and moment of force (torque) that act on this body in order to apply the laws of mechanics.

        **The free body diagram (FBD) of a mechanical system or model is the representation in a diagram of all forces and moments of force acting on each body, isolated from the rest of the system.**  

        The term free means that each body, which maybe was part of a connected system, is represented as isolated (free) and any existent contact force is represented in the diagram as forces (action and reaction) acting on the formerly connected bodies. Then, the laws of mechanics are applied on each body, and the unknown movement, force or moment of force can be found if the system of equations is determined (the number of unknown variables can not be greater than the number of equations for each body).

        How exactly a FBD is drawn for a mechanical model of something is dependent on what one is trying to find. For example, the air resistance might be neglected or not when modeling the movement of an object and the number of parts the system is divided is dependent on what is needed to know about the model.  

        The use of FBD is very common in biomechanics; a typical use is to use the FBD in order to determine the forces and torques on the ankle, knee, and hip joints of the lower limb (foot, leg, and thigh) during locomotion, and the FBD can be applied to any problem where the laws of mechanics are needed to solve a problem.

        For now, let's study how to draw free-body diagrams for systems that can be modeled as particles.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Steps to draw a free-body diagram (FBD)

        1. Draw separately each object considered in the problem. How you separate depends on what questions you want to answer.  
        2. Identify the forces acting on each object. If you are analyzing more than one object, remember the Newton's third Law (action and reaction), and identify where the reaction of a force is being applied.  
        3. Draw all the identified forces, representing them as vectors. The vectors should be represented with the origin in the object. In the case of particles, the origin should be in the center of the particle.  
        4. If necessary, you should represent the reference frame in the free-body diagram.  
        5. After this, you can solve the problem using the Newton's second Law (see, e.g, [Newton's Laws](https://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/Notebooks/newtonLawForParticles.ipynb)) to find the motion of the particle.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Basic element and forces
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Gravity

        The gravity force acts on two masses, each one atracting each other:$\vec{{\bf{F}}} = - G\frac{m_1m_2}{||\vec{\bf{r}}||^2}\frac{\vec{\bf{r}}}{||\vec{\bf{r}}||}$where$G = 6.67.10^{−11} Nm^2/kg^2$and$\vec{\bf{r}}$is a vector with length equal to the distance between the masses and directing towards the other mass. Note the forces acting on each mass have the same absolute value.

        Since the mass of the Earth is$m_1=5.9736×10^{24}kg$and its radius is 6.371×10$^6$m, the gravity force near the surface of the Earth is:

        <span class="notranslate">$\vec{{\bf{F}}} = m\vec{\bf{g}}$</span>

        with the absolute value of$\vec{\bf{g}}$approximately equal to 9.81$m/s^2$, pointing towards the center of Earth.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Spring

        Spring is an element used to represent a force proportional to some length or displacement. It produces a force in the same direction of the vector linking the spring extremities and opposite to its length or displacement from an equilibrium length. Frequently it has a linear relation, but it could be nonlinear as well. The force exerted by the spring in one of the extremities is: 

        <span class="notranslate">$\vec{\bf{F}} = - k(||\vec{\bf{r}}||-l_0)\frac{\vec{\bf{r}}}{||\vec{\bf{r}}||} = -k\vec{\bf{r}} +kl_0\frac{\vec{\bf{r}}}{||\vec{\bf{r}}||} = -k\left(1-\frac{l_0}{||\vec{\bf{r}}||}\right)\vec{\bf{r}}$</span>

        where$\vec{\bf{r}}$is the vector linking the extremity applying the force to the other extremity and$l_0$is the equilibrium length of the spring.

        Since the spring element is a massless element, the force in  both extremities have the same absolute value and opposite directions. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Damping 

        Damper is an element used to represent a force proportional to the velocity of displacement. It produces a force in the opposite direction of its velocity.

        Frequently it has a linear relation, but it could be nonlinear as well. The force exerted by the damper element in one of its extremities is: 

        <span class="notranslate">$\vec{\bf{F}} = - b||\vec{\bf{v}}||\frac{\vec{\bf{v}}}{||\vec{\bf{v}}||} = -b\vec{\bf{v}} = -b\frac{d\vec{\bf{r}}}{dt}$</span>

        where$\vec{\bf{r}}$is the vector linking the extremity applying the force to the other extremity.

        Since the damper element is a massless element, , the force in  both extremities have the same absolute value and opposite directions. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Examples of free-body diagram

        Let's see some examples on how to draw the free-body diagram and obtain the motion equations to solve the problems.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 1. No force acting on the particle

        The most trivial situation is a particle with no force acting on it.   

        The free-body diagram is below, with no force vectors acting on the particle.

        <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/ballNoGrav.png?raw=1" alt="free-body diagram of a ball" width="500"/><figcaption><i>Figure. Free-body diagram of a ball with no force acting on it.</i></figcaption></center></figure>
    
        In this situation, the resultant force is:

        <span class="notranslate">$\vec{\bf{F}} = 0$</span>
    
        And the second Newton law for this particle is:

        <span class="notranslate">$m\frac{d^2\vec{\bf{r}}}{dt^2} = 0 \quad \rightarrow \quad \frac{d^2\vec{\bf{r}}}{dt^2} = 0$</span>
    
        The motion of of the particle can be found by integrating twice both times, getting the following:

        <span class="notranslate">$\vec{\bf{r}} = \vec{\bf{v}}_0t + \vec{\bf{r}}_0$</span>
    
        The particle continues to change its position with the same velocity it was at the beginning of the analysis. This could be predicted by Newton's first law.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 2. Gravity force acting on the particle

        Now, let's consider a ball with the gravity force acting on it. The free-body diagram is depicted below.

        <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/ballGrav.png?raw=1" alt="free-body diagram of a ball" width="500"/><figcaption><i>Figure. Free-body diagram of a ball under the influence of gravity.</i></figcaption></center></figure>

        The only force acting on the ball is the gravitational force:

        <span class="notranslate">$\vec{\bf{F}}_g = - mg \; \hat{\bf{j}}$</span>

        Applying Newton's second Law:

        <span class="notranslate">$\vec{\bf{F}}_g = m \frac{d^2\vec{\bf{r}}}{dt^2} \rightarrow - mg \; \hat{\bf{j}} = m \frac{d^2\vec{\bf{r}}}{dt^2} \rightarrow - g \; \hat{\bf{j}} = \frac{d^2\vec{\bf{r}}}{dt^2}$</span>

        Now, we can separate the equation in two components (x and y):

        <span class="notranslate">$0 = \frac{d^2x}{dt^2}$</span>

        and

        <span class="notranslate">$- g = \frac{d^2y}{dt^2}$</span>

        These equations were solved in [this Notebook about the Newton's laws](https://nbviewer.jupyter.org/github/BMClab/BMC/blob/master/notebooks/newtonLawForParticles.ipynb). 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 3. Ground reaction force

        Now, we will analyze the situation of a particle at rest in contact with the ground. To simplify the analysis, only the vertical movement will be considered.

        <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/ballGravGRF.png?raw=1" alt="free-body diagram of a ball" width="400"/><figcaption><i>Figure. Free-body diagram of a ball at rest in contact with the ground.</i></figcaption></center></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The forces acting on the particle are the ground reaction force (often called as normal force) and the gravity force. The free-body diagram of the particle is below:

        <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/ballGravGRFFBD.png?raw=1" alt="free-body diagram of a ball" width="200"/><figcaption><i>Figure. Free-body diagram of a ball under the influence of gravity.</i></figcaption></center></figure>

        So, the resultant force in the particle is:
    
        <span class="notranslate">$\vec{\bf{F}} = \overrightarrow{\bf{GRF}} + m\vec{\bf{g}} = \overrightarrow{\bf{GRF}} - mg \; \hat{\bf{j}}$</span>

        Considering only the y direction:
    
        <span class="notranslate">$F = GRF - mg$</span>

        Applying Newton's second law to the particle:

        <span class="notranslate">$m \frac{d^2y}{dt^2} = GRF - mg$</span>

        Note that since we have no information about how the force GRF varies along time, we cannot solve this equation. To find the position of the particle along time, one would have to measure the ground reaction force. See [the notebook on Vertical jump](http://nbviewer.jupyter.org/github/BMClab/BMC/blob/master/notebooks/VerticalJump.ipynb) for an application of this model.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 4. Mass-spring system with horizontal movement

        The example below represents a mass attached to a spring and the other extremity of the spring is fixed. 

        <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/ballspring.png?raw=1" alt="free-body diagram of a ball" width="500"/><figcaption><i>Figure. Mass-spring system with horizontal movement.</i></figcaption></center></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The only force force acting on the mass is from the spring. Below is the free-body diagram from the mass.

        <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/ballspringFBD.png?raw=1" alt="free-body diagram of a ball" width="200"/><figcaption><i>Figure. Free-body diagram of a mass-spring system.</i></figcaption></center></figure>

        Since the movement is horizontal, we can neglect the gravity force.  

        <span class="notranslate">$\vec{\bf{F}} = -k\left(1-\frac{l_0}{||\vec{\bf{r}}||}\right)\vec{\bf{r}}$</span>
    
        Applying Newton's second law to the mass:
    
        <span class="notranslate">$m\frac{d^2\vec{\bf{r}}}{dt^2} = -k\left(1-\frac{l_0}{||\vec{\bf{r}}||}\right)\vec{\bf{r}} \rightarrow \frac{d^2\vec{\bf{r}}}{dt^2} = -\frac{k}{m}\left(1-\frac{l_0}{||\vec{\bf{r}}||}\right)\vec{\bf{r}}$</span>
    
        Since the movement is unidimensional, we can deal with it scalarly:
    
        <span class="notranslate">$\frac{d^2x}{dt^2} = -\frac{k}{m}\left(1-\frac{l_0}{x}\right)x = -\frac{k}{m}(x-l_0)$</span>

        To solve this equation numerically, we must break the equations into two first-order differential equation:
    
        <span class="notranslate">$\frac{dv_x}{dt} =  -\frac{k}{m}(x-l_0)$</span>

        <span class="notranslate">$\frac{dx}{dt} =  v_x$</span>

        In the numerical solution below, we will use$k = 40 N/m$,$m = 2 kg$,$l_0 = 0.5 m$and the mass starts from the position$x = 0.8m$and at rest.
        """
    )
    return


@app.cell
def _(np, plt):
    _k = 40
    _m = 2
    _l0 = 0.5
    x0 = 0.8
    v0 = 0
    x = x0
    v = v0
    _dt = 0.001
    _t = np.arange(0, 3, _dt)
    r = np.array([x])
    for _i in _t[1:]:
        dxdt = v
        dvxdt = -_k / _m * (x - _l0)
        x = x + _dt * dxdt
        v = v + _dt * dvxdt
        r = np.vstack((r, np.array([x])))
    plt.figure(figsize=(8, 4))
    plt.plot(_t, r, lw=4)
    plt.xlabel('t(s)')
    plt.ylabel('x(m)')
    plt.title('Spring displacement')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 5. Linear spring in bidimensional movement at horizontal plane

        This example below represents a system with two masses attached to a spring.  
        To solve the motion of both masses, we have to draw a free-body diagram for each one of the masses. 

        <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/twoballspring.png?raw=1" alt="Linear spring" width="500"/><figcaption><i>Figure. Linear spring in bidimensional movement at horizontal plane.</i></figcaption></center></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The only force acting on each mass is the force due to the spring. Since the movement is happening at the horizontal plane, the gravity force can be neglected.

        <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/twoballspringFBD.png?raw=1" alt="Linear spring" width="200"/><figcaption><i>Figure. FBD of linear spring in bidimensional movement at horizontal plane.</i></figcaption></center></figure>

        So, the forces acting on mass 1 is:

        <span class="notranslate">$\vec{\bf{F_1}} = k\left(||\vec{\bf{r_2}}-\vec{\bf{r_1}}||-l_0\right)\frac{(\vec{\bf{r_2}}-\vec{\bf{r_1}})}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}$</span>
    
        and the forces acting on mass 2 is:
    
        <span class="notranslate">$\vec{\bf{F_2}} =k\left(||\vec{\bf{r_2}}-\vec{\bf{r_1}}||-l_0\right)\frac{(\vec{\bf{r_1}}-\vec{\bf{r_2}})}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Applying Newton's second law for the masses:

        <span class="notranslate">$m_1\frac{d^2\vec{\bf{r_1}}}{dt^2} = k\left(||\vec{\bf{r_2}}-\vec{\bf{r_1}}||-l_0\right)\frac{(\vec{\bf{r_2}}-\vec{\bf{r_1}})}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||} 
            \\
            \frac{d^2\vec{\bf{r_1}}}{dt^2} = -\frac{k}{m_1}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)\vec{\bf{r_1}}+\frac{k}{m_1}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)\vec{\bf{r_2}} 
            \\
            \frac{d^2x_1\hat{\bf{i}}}{dt^2}+\frac{d^2y_1\hat{\bf{j}}}{dt^2} = -\frac{k}{m_1}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)(x_1\hat{\bf{i}}+y_1\hat{\bf{j}})+\frac{k}{m_1}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)(x_2\hat{\bf{i}}+y_2\hat{\bf{j}})$</span>

        <br/>

        <span class="notranslate">$m_2\frac{d^2\vec{\bf{r_2}}}{dt^2} = k\left(||\vec{\bf{r_2}}-\vec{\bf{r_1}}||-l_0\right)\frac{(\vec{\bf{r_1}}-\vec{\bf{r_2}})}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}
            \\
            \frac{d^2\vec{\bf{r_2}}}{dt^2} = -\frac{k}{m_2}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)\vec{\bf{r_2}}+\frac{k}{m_2}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)\vec{\bf{r_1}} 
            \\
            \frac{d^2x_2\hat{\bf{i}}}{dt^2}+\frac{d^2y_2\hat{\bf{j}}}{dt^2} = -\frac{k}{m_2}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)(x_2\hat{\bf{i}}+y_2\hat{\bf{j}})+\frac{k}{m_2}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)(x_1\hat{\bf{i}}+y_1\hat{\bf{j}})$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now, we can separate the equations for each of the coordinates:

        <span class="notranslate">$\frac{d^2x_1}{dt^2} = -\frac{k}{m_1}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)x_1+\frac{k}{m_1}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)x_2=-\frac{k}{m_1}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(x_1-x_2)$</span>

        <span class="notranslate">$\frac{d^2y_1}{dt^2} = -\frac{k}{m_1}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)y_1+\frac{k}{m_1}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)y_2=-\frac{k}{m_1}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(y_1-y_2)$</span>

        <span class="notranslate">$\frac{d^2x_2}{dt^2} = -\frac{k}{m_2}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)x_2+\frac{k}{m_2}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)x_1=-\frac{k}{m_2}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(x_2-x_1)$</span>
        
        <span class="notranslate">$\frac{d^2y_2}{dt^2} = -\frac{k}{m_2}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)y_2+\frac{k}{m_2}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)y_1=-\frac{k}{m_2}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(y_2-y_1)$</span>    
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To solve these equations numerically, you must break these equations into first-order equations:

        <span class="notranslate">$\frac{dv_{x_1}}{dt} = -\frac{k}{m_1}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(x_1-x_2)$</span>    

        <span class="notranslate">$\frac{dv_{y_1}}{dt} = -\frac{k}{m_1}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(y_1-y_2)$</span>    

        <span class="notranslate">$\frac{dv_{x_2}}{dt} = -\frac{k}{m_2}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(x_2-x_1)$</span>    

        <span class="notranslate">$\frac{dv_{y_2}}{dt} = -\frac{k}{m_2}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(y_2-y_1)$</span>    

        <span class="notranslate">$\frac{dx_1}{dt} = v_{x_1}$</span>    

        <span class="notranslate">$\frac{dy_1}{dt} = v_{y_1}$</span>    

        <span class="notranslate">$\frac{dx_2}{dt} = v_{x_2}$</span>    

        <span class="notranslate">$\frac{dy_2}{dt} = v_{y_2}$</span>    
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that if you did not want to know the details about the motion of each mass, but only the motion of the center of mass of the masses-spring system, you could have modeled the whole system as a single particle.

        To solve the equations numerically, we will use the$m_1=1 kg$,$m_2 = 2 kg$,$l_0 = 0.5 m$,$k = 90 N/m$and$x_{1_0} = 0 m$,$x_{2_0} = 0 m$,$y_{1_0} = 1 m$,$y_{2_0} = -1 m$,$v_{x1_0} = -2 m/s$,$v_{x2_0} = 0 m/s$,$v_{y1_0} = 0 m/s$,$v_{y2_0} = 0 m/s$. 
        """
    )
    return


@app.cell
def _(np, plt):
    _x01 = 0
    _y01 = 0.5
    _x02 = 0
    _y02 = -0.5
    _vx01 = 0.1
    _vy01 = 0
    _vx02 = -0.1
    _vy02 = 0
    _x1 = _x01
    _y1 = _y01
    _x2 = _x02
    _y2 = _y02
    _vx1 = _vx01
    _vy1 = _vy01
    _vx2 = _vx02
    _vy2 = _vy02
    _r1 = np.array([_x1, _y1])
    _r2 = np.array([_x2, _y2])
    _k = 30
    _m1 = 1
    _m2 = 1
    _l0 = 0.5
    _dt = 0.0001
    _t = np.arange(0, 5, _dt)
    for _i in _t[1:]:
        _dvx1dt = -_k / _m1 * (_x1 - _x2) * (1 - _l0 / np.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2))
        _dvx2dt = -_k / _m2 * (_x2 - _x1) * (1 - _l0 / np.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2))
        _dvy1dt = -_k / _m1 * (_y1 - _y2) * (1 - _l0 / np.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2))
        _dvy2dt = -_k / _m2 * (_y2 - _y1) * (1 - _l0 / np.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2))
        _dx1dt = _vx1
        _dx2dt = _vx2
        _dy1dt = _vy1
        _dy2dt = _vy2
        _x1 = _x1 + _dt * _dx1dt
        _x2 = _x2 + _dt * _dx2dt
        _y1 = _y1 + _dt * _dy1dt
        _y2 = _y2 + _dt * _dy2dt
        _vx1 = _vx1 + _dt * _dvx1dt
        _vx2 = _vx2 + _dt * _dvx2dt
        _vy1 = _vy1 + _dt * _dvy1dt
        _vy2 = _vy2 + _dt * _dvy2dt
        _r1 = np.vstack((_r1, np.array([_x1, _y1])))
        _r2 = np.vstack((_r2, np.array([_x2, _y2])))
    springLength = np.sqrt((_r1[:, 0] - _r2[:, 0]) ** 2 + (_r1[:, 1] - _r2[:, 1]) ** 2)
    plt.figure(figsize=(8, 4))
    plt.plot(_t, springLength, lw=4)
    plt.xlabel('t(s)')
    plt.ylabel('Spring length (m)')
    plt.show()
    plt.figure(figsize=(8, 4))
    plt.plot(_r1[:, 0], _r1[:, 1], 'r.', lw=4)
    plt.plot(_r2[:, 0], _r2[:, 1], 'b.', lw=4)
    plt.plot((_m1 * _r1[:, 0] + _m2 * _r2[:, 0]) / (_m1 + _m2), (_m1 * _r1[:, 1] + _m2 * _r2[:, 1]) / (_m1 + _m2), 'g.')
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.7, 0.7)
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.title('Masses position')
    plt.legend(('Mass1', 'Mass 2', 'Masses center of mass'))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 6. Particle under action of gravity and linear air resistance

        Below is the free-body diagram of a particle with the gravity force and a linear drag force due to the air resistance. 

        <figure><center><img src="../images/ballGravLinearRes.png" alt="Linear spring" width="700"/><figcaption><i>Figure. Particle under action of gravity and linear air resistance.</i></figcaption></center></figure>  

        the forces being applied in the ball are:

        <span class="notranslate">$\vec{\bf{F}} = -mg \hat{\bf{j}} - b\vec{\bf{v}} = -mg \hat{\bf{j}} - b\frac{d\vec{\bf{r}}}{dt} = -mg \hat{\bf{j}} - b\left(\frac{dx}{dt}\hat{\bf{i}}+\frac{dy}{dt}\hat{\bf{j}}\right) = - b\frac{dx}{dt}\hat{\bf{i}} - \left(mg + b\frac{dy}{dt}\right)\hat{\bf{j}}$</span>

        Writing down Newton's second law:

        <span class="notranslate">$\vec{\bf{F}} = m \frac{d^2\vec{\bf{r}}}{dt^2} \rightarrow - b\frac{dx}{dt}\hat{\bf{i}} - \left(mg + b\frac{dy}{dt}\right)\hat{\bf{j}} = m\left(\frac{d^2x}{dt^2}\hat{\bf{i}}+\frac{d^2y}{dt^2}\hat{\bf{j}}\right)$</span>

        Now, we can separate into one equation for each coordinate:

        <span class="notranslate">$- b\frac{dx}{dt} = m\frac{d^2x}{dt^2} -\rightarrow \frac{d^2x}{dt^2} = -\frac{b}{m} \frac{dx}{dt}$</span>

        <span class="notranslate">$-mg - b\frac{dy}{dt} = m\frac{d^2y}{dt^2} \rightarrow \frac{d^2y}{dt^2} = -\frac{b}{m}\frac{dy}{dt} - g$</span>

        These equations were solved in [this notebook](https://nbviewer.jupyter.org/github/BMClab/BMC/blob/master/notebooks/newtonLawForParticles.ipynb).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 7. Particle under action of gravity and nonlinear air resistance

        Below, is the free-body diagram of a particle with the gravity force and a drag force due to the air resistance proportional to the square of the particle velocity. 

        <figure><center><img src="../images/ballGravSquareRes.png" alt="Linear spring" width="700"/><figcaption><i>Figure. Particle under action of gravity and nonlinear air resistance.</i></figcaption></center></figure> 
    
        The forces being applied in the ball are:

        <span class="notranslate">$\vec{\bf{F}} = -mg \hat{\bf{j}} - bv^2\hat{\bf{e_t}} = -mg \hat{\bf{j}} - b (v_x^2+v_y^2) \frac{v_x\hat{\bf{i}}+v_y\hat{\bf{j}}}{\sqrt{v_x^2+v_y^2}} =   -mg \hat{\bf{j}} - b \sqrt{v_x^2+v_y^2} \,(v_x\hat{\bf{i}}+v_y\hat{\bf{j}}) =  -mg \hat{\bf{j}} - b \sqrt{\left(\frac{dx}{dt} \right)^2+\left(\frac{dy}{dt} \right)^2} \,\left(\frac{dx}{dt} \hat{\bf{i}}+\frac{dy}{dt}\hat{\bf{j}}\right)$</span>

        Writing down Newton's second law:

        <span class="notranslate">$\vec{\bf{F}} = m \frac{d^2\vec{\bf{r}}}{dt^2} \rightarrow -mg \hat{\bf{j}} - b \sqrt{\left(\frac{dx}{dt} \right)^2+\left(\frac{dy}{dt} \right)^2} \,\left(\frac{dx}{dt} \hat{\bf{i}}+\frac{dy}{dt}\hat{\bf{j}}\right) = m\left(\frac{d^2x}{dt^2}\hat{\bf{i}}+\frac{d^2y}{dt^2}\hat{\bf{j}}\right)$</span>

        Now, we can separate into one equation for each coordinate:

        <span class="notranslate">$- b \sqrt{\left(\frac{dx}{dt} \right)^2+\left(\frac{dy}{dt} \right)^2} \,\frac{dx}{dt} = m\frac{d^2x}{dt^2} \rightarrow \frac{d^2x}{dt^2} = - \frac{b}{m} \sqrt{\left(\frac{dx}{dt} \right)^2+\left(\frac{dy}{dt} \right)^2} \,\frac{dx}{dt}$</span>

        <span class="notranslate">$-mg - b \sqrt{\left(\frac{dx}{dt} \right)^2+\left(\frac{dy}{dt} \right)^2} \,\frac{dy}{dt} = m\frac{d^2y}{dt^2} \rightarrow \frac{d^2y}{dt^2} = - \frac{b}{m} \sqrt{\left(\frac{dx}{dt} \right)^2+\left(\frac{dy}{dt} \right)^2} \,\frac{dy}{dt} -g$</span>

        These equations were solved numerically in [this notebook](https://nbviewer.jupyter.org/github/BMClab/BMC/blob/master/notebooks/newtonLawForParticles.ipynb).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 8. Linear spring and damping on bidimensional horizontal movement

        This situation is very similar to the example of horizontal movement with one spring and two masses, with a damper added in parallel to the spring.

        <figure><center><img src="../images/twomassDamp.png" alt="Linear spring"/><figcaption><i>Figure. Linear spring and damping on bidimensional horizontal movement.</i></figcaption></center></figure>    
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now, the forces acting on each mass are the force due to the spring and the force due to the damper. 

        <figure><center><img src="../images/twomassDampFBD.png" alt="Linear spring" width="200"/><figcaption><i>Figure. FBD of linear spring and damping on bidimensional horizontal movement.</i></figcaption></center></figure>    

        So, the forces acting on mass 1 is:

        <span class="notranslate">$\vec{\bf{F_1}} = b\frac{d(\vec{\bf{r_2}}-\vec{\bf{r_1}})}{dt} +  k\left(||\vec{\bf{r_2}}-\vec{\bf{r_1}}||-l_0\right)\frac{(\vec{\bf{r_2}}-\vec{\bf{r_1}})}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||} =  b\frac{d(\vec{\bf{r_2}}-\vec{\bf{r_1}})}{dt} +  k\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)(\vec{\bf{r_2}}-\vec{\bf{r_1}})$</span>

        and the forces acting on mass 2 is:

        <span class="notranslate">$\vec{\bf{F_2}} = b\frac{d(\vec{\bf{r_1}}-\vec{\bf{r_2}})}{dt} +  k\left(||\vec{\bf{r_2}}-\vec{\bf{r_1}}||-l_0\right)\frac{(\vec{\bf{r_1}}-\vec{\bf{r_2}})}{||\vec{\bf{r_1}}-\vec{\bf{r_2}}||}= b\frac{d(\vec{\bf{r_1}}-\vec{\bf{r_2}})}{dt} +  k\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)(\vec{\bf{r_1}}-\vec{\bf{r_2}})$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Applying the Newton's second law for the masses:

        <span class="notranslate">$m_1\frac{d^2\vec{\bf{r_1}}}{dt^2} = b\frac{d(\vec{\bf{r_2}}-\vec{\bf{r_1}})}{dt}+k\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)(\vec{\bf{r_2}}-\vec{\bf{r_1}})$</span>

        <span class="notranslate">$\frac{d^2\vec{\bf{r_1}}}{dt^2} = -\frac{b}{m_1}\frac{d\vec{\bf{r_1}}}{dt} -\frac{k}{m_1}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)\vec{\bf{r_1}} + \frac{b}{m_1}\frac{d\vec{\bf{r_2}}}{dt}+\frac{k}{m_1}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)\vec{\bf{r_2}}$</span>

        <span class="notranslate">$\frac{d^2x_1\hat{\bf{i}}}{dt^2}+\frac{d^2y_1\hat{\bf{j}}}{dt^2} = -\frac{b}{m_1}\left(\frac{dx_1\hat{\bf{i}}}{dt}+\frac{dy_1\hat{\bf{j}}}{dt}\right)-\frac{k}{m_1}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)(x_1\hat{\bf{i}}+y_1\hat{\bf{j}})+\frac{b}{m_1}\left(\frac{dx_2\hat{\bf{i}}}{dt}+\frac{dy_2\hat{\bf{j}}}{dt}\right)+\frac{k}{m_1}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)(x_2\hat{\bf{i}}+y_2\hat{\bf{j}}) = -\frac{b}{m_1}\left(\frac{dx_1\hat{\bf{i}}}{dt}+\frac{dy_1\hat{\bf{j}}}{dt}-\frac{dx_2\hat{\bf{i}}}{dt}-\frac{dy_2\hat{\bf{j}}}{dt}\right)-\frac{k}{m_1}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(x_1\hat{\bf{i}}+y_1\hat{\bf{j}}-x_2\hat{\bf{i}}-y_2\hat{\bf{j}})$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""$m_2\frac{d^2\vec{\bf{r_2}}}{dt^2} = b\frac{d(\vec{\bf{r_1}}-\vec{\bf{r_2}})}{dt}+k\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)(\vec{\bf{r_1}}-\vec{\bf{r_2}})$$\frac{d^2\vec{\bf{r_2}}}{dt^2} = -\frac{b}{m_2}\frac{d\vec{\bf{r_2}}}{dt} -\frac{k}{m_2}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)\vec{\bf{r_2}} + \frac{b}{m_2}\frac{d\vec{\bf{r_1}}}{dt}+\frac{k}{m_2}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)\vec{\bf{r_1}}$$\frac{d^2x_2\hat{\bf{i}}}{dt^2}+\frac{d^2y_2\hat{\bf{j}}}{dt^2} = -\frac{b}{m_2}\left(\frac{dx_2\hat{\bf{i}}}{dt}+\frac{dy_2\hat{\bf{j}}}{dt}\right)-\frac{k}{m_2}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)(x_2\hat{\bf{i}}+y_2\hat{\bf{j}})+\frac{b}{m_2}\left(\frac{dx_1\hat{\bf{i}}}{dt}+\frac{dy_1\hat{\bf{j}}}{dt}\right)+\frac{k}{m_2}\left(1-\frac{l_0}{||\vec{\bf{r_2}}-\vec{\bf{r_1}}||}\right)(x_1\hat{\bf{i}}+y_1\hat{\bf{j}})=-\frac{b}{m_2}\left(\frac{dx_2\hat{\bf{i}}}{dt}+\frac{dy_2\hat{\bf{j}}}{dt}-\frac{dx_1\hat{\bf{i}}}{dt}-\frac{dy_1\hat{\bf{j}}}{dt}\right)-\frac{k}{m_2}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(x_2\hat{\bf{i}}+y_2\hat{\bf{j}}-x_1\hat{\bf{i}}-y_1\hat{\bf{j}})$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now, we can separate the equations for each of the coordinates:

        <span class="notranslate">$\frac{d^2x_1}{dt^2} = -\frac{b}{m_1}\left(\frac{dx_1}{dt}-\frac{dx_2}{dt}\right)-\frac{k}{m_1}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(x_1-x_2)$</span>

        <span class="notranslate">$\frac{d^2y_1}{dt^2} = -\frac{b}{m_1}\left(\frac{dy_1}{dt}-\frac{dy_2}{dt}\right)-\frac{k}{m_1}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(y_1-y_2)$</span>

        <span class="notranslate">$\frac{d^2x_2}{dt^2} = -\frac{b}{m_2}\left(\frac{dx_2}{dt}-\frac{dx_1}{dt}\right)-\frac{k}{m_2}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(x_2-x_1)$</span>

        <span class="notranslate">$\frac{d^2y_2}{dt^2} = -\frac{b}{m_2}\left(\frac{dy_2}{dt}-\frac{dy_1}{dt}\right)-\frac{k}{m_2}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(y_2-y_1)$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you want to solve these equations numerically, you must break these equations into first-order  equations:

        <span class="notranslate">$\frac{dv_{x_1}}{dt} = -\frac{b}{m_1}\left(v_{x_1}-v_{x_2}\right)-\frac{k}{m_1}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(x_1-x_2)$</span>

        <span class="notranslate">$\frac{dv_{y_1}}{dt} =  -\frac{b}{m_1}\left(v_{y_1}-v_{y_2}\right)-\frac{k}{m_1}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(y_1-y_2)$</span>

        <span class="notranslate">$\frac{dv_{x_2}}{dt} = -\frac{b}{m_2}\left(v_{x_2}-v_{x_1}\right)-\frac{k}{m_2}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(x_2-x_1)$</span>

        <span class="notranslate">$\frac{dv_{y_2}}{dt} = -\frac{b}{m_2}\left(v_{y_2}-v_{y_1}\right)-\frac{k}{m_2}\left(1-\frac{l_0}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right)(y_2-y_1)$</span>

        <span class="notranslate">$\frac{dx_1}{dt} = v_{x_1}$</span>

        <span class="notranslate">$\frac{dy_1}{dt} = v_{y_1}$</span>

        <span class="notranslate">$\frac{dx_2}{dt} = v_{x_2}$</span>

        <span class="notranslate">$\frac{dy_2}{dt} = v_{y_2}$</span>

        To solve the equations numerically, we will use the$m_1=1 kg$,$m_2 = 2 kg$,$l_0 = 0.5 m$,$k = 10 N/m$,$b = 0.6 Ns/m$and$x_{1_0} = 0 m$,$x_{2_0} = 0 m$,$y_{1_0} = 1 m$,$y_{2_0} = -1 m$,$v_{x1_0} = -2 m/s$,$v_{x2_0} = 1 m/s$,$v_{y1_0} = 0 m/s$,$v_{y2_0} = 0 m/s$. 
        """
    )
    return


@app.cell
def _(np, plt):
    _x01 = 0
    _y01 = 1
    _x02 = 0
    _y02 = -1
    _vx01 = -2
    _vy01 = 0
    _vx02 = 1
    _vy02 = 0
    _x1 = _x01
    _y1 = _y01
    _x2 = _x02
    _y2 = _y02
    _vx1 = _vx01
    _vy1 = _vy01
    _vx2 = _vx02
    _vy2 = _vy02
    _r1 = np.array([_x1, _y1])
    _r2 = np.array([_x2, _y2])
    _k = 10
    _m1 = 1
    _m2 = 2
    _b = 0.6
    _l0 = 0.5
    _dt = 0.001
    _t = np.arange(0, 5, _dt)
    for _i in _t[1:]:
        _dvx1dt = -_b / _m1 * (_vx1 - _vx2) - _k / _m1 * (1 - _l0 / np.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2)) * (_x1 - _x2)
        _dvx2dt = -_b / _m2 * (_vx2 - _vx1) - _k / _m2 * (1 - _l0 / np.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2)) * (_x2 - _x1)
        _dvy1dt = -_b / _m1 * (_vy1 - _vy2) - _k / _m1 * (1 - _l0 / np.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2)) * (_y1 - _y2)
        _dvy2dt = -_b / _m2 * (_vy2 - _vy1) - _k / _m2 * (1 - _l0 / np.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2)) * (_y2 - _y1)
        _dx1dt = _vx1
        _dx2dt = _vx2
        _dy1dt = _vy1
        _dy2dt = _vy2
        _x1 = _x1 + _dt * _dx1dt
        _x2 = _x2 + _dt * _dx2dt
        _y1 = _y1 + _dt * _dy1dt
        _y2 = _y2 + _dt * _dy2dt
        _vx1 = _vx1 + _dt * _dvx1dt
        _vx2 = _vx2 + _dt * _dvx2dt
        _vy1 = _vy1 + _dt * _dvy1dt
        _vy2 = _vy2 + _dt * _dvy2dt
        _r1 = np.vstack((_r1, np.array([_x1, _y1])))
        _r2 = np.vstack((_r2, np.array([_x2, _y2])))
    springDampLength = np.sqrt((_r1[:, 0] - _r2[:, 0]) ** 2 + (_r1[:, 1] - _r2[:, 1]) ** 2)
    plt.figure(figsize=(8, 4))
    plt.plot(_t, springDampLength, lw=4)
    plt.xlabel('t(s)')
    plt.ylabel('Spring length (m)')
    plt.show()
    plt.figure(figsize=(8, 4))
    plt.plot(_r1[:, 0], _r1[:, 1], 'r.', lw=4)
    plt.plot(_r2[:, 0], _r2[:, 1], 'b.', lw=4)
    plt.plot((_m1 * _r1[:, 0] + _m2 * _r2[:, 0]) / (_m1 + _m2), (_m1 * _r1[:, 1] + _m2 * _r2[:, 1]) / (_m1 + _m2), 'g.')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.title('Masses position')
    plt.legend(('Mass1', 'Mass 2', 'Masses center of mass'))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 9. Simple muscle model

        The diagram below shows a simple muscle model. The spring in the left represents the tendinous tissues and the spring in the right represents the elastic properties of the muscle fibers. The damping is present to model the viscous properties of the muscle fibers, the element CE is the contractile element (force production) and the mass$m$is the muscle mass. 

        The length$L_{MT}$is the length of the muscle plus the tendon. In our model$L_{MT}$is constant, but it could be a function of the joint angle. 

        <figure><center><img src="../images/simpleMuscle.png" alt="Linear spring" width="500"/><figcaption><i>Figure. Simple muscle model.</i></figcaption></center></figure>    

        The length of the tendon will be denoted by$l_T(t)$and the muscle length, by$l_{m}(t)$. Both lengths are related by each other  by the following expression:

        <span class="notranslate">$l_t(t) + l_m(t) = L_{MT}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The free-body diagram of the muscle mass is depicted below.

        <figure><center><img src="../images/simpleMuscleFBD.png" alt="Linear spring" width="200"/><figcaption><i>Figure. FBD of simple muscle model.</i></figcaption></center></figure>    
    
        The resultant force being applied in the muscle mass is:

        <span class="notranslate">$\vec{\bf{F}} = -k_T(||\vec{\bf{r_m}}||-l_{t_0})\frac{\vec{\bf{r_m}}}{||\vec{\bf{r_m}}||} + b\frac{d(L_{MT}\hat{\bf{i}} - \vec{\bf{r_{m}}})}{dt} + k_m (||L_{MT}\hat{\bf{i}} - \vec{\bf{r_{m}}}||-l_{m_0})\frac{L_{MT}\hat{\bf{i}} - \vec{\bf{r_{m}}}}{||L_{MT}\hat{\bf{i}} - \vec{\bf{r_{m}}}||} +\vec{\bf{F}}{\bf{_{CE}}}(t)$</span>

        where$\vec{\bf{r_m}}$is the muscle mass position.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since the model is unidimensional, we can assume that the force$\vec{\bf{F}}\bf{_{CE}}(t)$is in the x direction, so the analysis will be done only in this direction. 

        <span class="notranslate">$F = -k_T(l_t-l_{t_0}) + b\frac{d(L_{MT} - l_t)}{dt} + k_m (l_m-l_{m_0}) + F_{CE}(t) \\
        F = -k_T(l_t-l_{t_0}) -b\frac{dl_t}{dt} + k_m (L_{MT}-l_t-l_{m_0}) + F_{CE}(t) \\
        F = -b\frac{dl_t}{dt}-(k_T+k_m)l_t+F_{CE}(t)+k_Tl_{t_0}+k_m(L_{MT}-l_{m_0})$</span>

        Applying the Newton's second law:

        <span class="notranslate">$m\frac{d^2l_t}{dt^2} = -b\frac{dl_t}{dt}-(k_T+k_m)l_t+F_{CE}(t)+k_Tl_{t_0}+k_m(L_{MT}-l_{m_0})$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To solve this equation, we must break the equation into two first-order differential equations:

        <span class="notranslate">$\frac{dvt}{dt} = - \frac{b}{m}v_t - \frac{k_T+k_m}{m}l_t +\frac{F_{CE}(t)}{m} + \frac{k_T}{m}l_{t_0}+\frac{k_m}{m}(L_{MT}-l_{m_0})$</span>

        <span class="notranslate">$\frac{d l_t}{dt} = v_t$</span>

        Now, we can solve these equations by using some numerical method. To obtain the solution, we will use the damping factor of the muscle as$b = 10\,Ns/m$, the muscle mass is$m = 2 kg$, the stiffness of the tendon as$k_t=1000\,N/m$and the elastic element of the muscle as$k_m=1500\,N/m$. The tendon-length is$L_{MT} = 0.35\,m$, and the tendon equilibrium length is$l_{t0} = 0.28\,m$and the muscle fiber equilibrium length is$l_{m0} = 0.07\,m$. Both the tendon and the muscle fiber are at their equilibrium lengths and at rest.

        Also, we will model the force of the contractile element as a Heaviside step of$90\,N$(90 N beginning at$t=0$), but normally it is modeled as a function of$l_m$and$v_m$having a neural activation signal as input. 
        """
    )
    return


@app.cell
def _(np, plt):
    _m = 2
    _b = 10
    km = 1500
    kt = 1000
    lt0 = 0.28
    lm0 = 0.07
    Lmt = 0.35
    vt0 = 0
    _dt = 0.0001
    _t = np.arange(0, 10, _dt)
    Fce = 90
    lt = lt0
    vt = vt0
    ltp = np.array([lt0])
    lmp = np.array([lm0])
    Ft = np.array([0])
    for _i in range(1, len(_t)):
        dvtdt = -_b / _m * vt - (kt + km) / _m * lt + Fce / _m + kt / _m * lt0 + km / _m * (Lmt - lm0)
        dltdt = vt
        vt = vt + _dt * dvtdt
        lt = lt + _dt * dltdt
        Ft = np.vstack((Ft, np.array(kt * (lt - lt0))))
        ltp = np.vstack((ltp, np.array(lt)))
        lmp = np.vstack((lmp, np.array(Lmt - lt)))
    plt.figure(figsize=(8, 4))
    plt.plot(_t, Ft, lw=4)
    plt.xlabel('t(s)')
    plt.ylabel('Tendon force (N)')
    plt.show()
    plt.figure(figsize=(8, 4))
    plt.plot(_t, ltp, lw=4)
    plt.plot(_t, lmp, lw=4)
    plt.xlabel('t(s)')
    plt.ylabel('Length (m)')
    plt.legend(('Tendon length', 'Muscle fiber length'))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

        - Read the 2nd chapter of the [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html) about free-body diagrams;
        - Read the 13th of the [Hibbeler's book](https://drive.google.com/file/d/1sDLluWCiBCog2C11_Iu1fjv-BtfVUxBU/view) (available in the Classroom).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. Draw free-body diagram for (see answers at https://youtu.be/3rZR7FSSidc):
           1. A book is at rest on a table.  
           2. A book is attached to a string and hanging from the ceiling.  
           3. A person pushes a crate to the right across a floor at a constant speed.  
           4. A skydiver is falling downward a speeding up.  
           5. A rightward-moving car has locked wheels and is skidding to a stop.  
           6. A freight elevator is attached by a cable, being pulled upward, and slowing down. It is not touching the sides of the elevator shaft.
   
        2. A block (2) is stacked on top of block (1) which is at rest on a surface (all contacts have friction). Which is the maximum horizontal force that can be applied to the block (1) such that block (2) does not slip? Answer at https://youtu.be/63U4_OxohOw

        3. Consider two masses m1 = 5 kg and m2 = 3 kg attached to a pulley which is attached to a wall with a rope and m1 is pulled in the opposite direction with force F = 100 N as shown in the figure below. Calculate (Answer at https://youtu.be/Dgg76vNChEU):
           1. Friction force on m1
           2. Friction force on m2
           3. Tension on the rope attached to masses m1 and m2
           4. Acceleration of m1  
        <figure><img src="./../images/friction_block.png" width="300"/></figure>


        4. (Example 13.4 of Hibbeler's book) A smooth 2-kg collar C, shown in the figure below, is attached to a spring having a stiffness k = 3 N/m and an unstretched length of 0.75 m. If the collar is released from rest at A, determine its acceleration and the normal force of the rod on the collar at the instant y = 1 m.
        <figure><img src="./../images/spring_collar.png" width="200"/></figure>

        5. (Example 13.5 of Hibbeler's book) The 100-kg block A shown in the figure below is released from rest. If the masses of the pulleys and the cord are neglected, determine the speed of the 20-kg block B in 2 s.
        <figure><img src="./../images/pulley_block.png" width="200"/></figure>

        6. (Example 13.9 of Hibbeler's book) The 60-kg skateboarder in the figure below coasts down the circular track. If he starts from rest when e = 0$^o$, determine the magnitude of the normal reaction the track exerts on him when e = 60$^o$. Neglect his size for the calculation.
        <figure><img src="./../images/skateboarder.png" width="300"/></figure>

        7. Solve the problems 2.3.9, 2.3.20, 11.1.6, 13.1.6 (a, b, c, d, f), 13.1.7, 13.1.10 (a, b) from Ruina and Pratap's book.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Ruina A, Rudra P (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press. 
        - R. C. Hibbeler (2010) [Engineering Mechanics Dynamics](https://drive.google.com/file/d/1sDLluWCiBCog2C11_Iu1fjv-BtfVUxBU/view). 12th Edition. Pearson Prentice Hall.
        - Nigg & Herzog (2006) [Biomechanics of the Musculo-skeletal System](https://books.google.com.br/books?id=hOIeAQAAIAAJ&dq=editions:ISBN0470017678). 3rd Edition. Wiley.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
