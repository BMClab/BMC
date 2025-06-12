import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Kinetics: fundamental concepts

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
        Kinetics is the branch of classical mechanics that is concerned with the relationship between the motion of bodies and its causes, namely forces and torques ([Encyclopædia Britannica Online](https://www.britannica.com/science/kinetics)).  
        Kinetics, as used in Biomechanics, also includes the study of statics, the study of equilibrium and its relation to forces and torques (one can treat equilibrium as a special case of motion, where the velocity is zero). This is different than the nowadays most common ramification of Mechanics in Statics and Dynamics, and Dynamics in Kinematics and Kinetics ([Introduction to Biomechanics](http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Biomechanics.ipynb#On-the-branches-of-Mechanics-and-Biomechanics-I)).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Python-setup" data-toc-modified-id="Python-setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Python setup</a></span></li><li><span><a href="#The-development-of-the-laws-of-motion-of-bodies" data-toc-modified-id="The-development-of-the-laws-of-motion-of-bodies-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>The development of the laws of motion of bodies</a></span></li><li><span><a href="#Newton's-laws-of-motion" data-toc-modified-id="Newton's-laws-of-motion-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Newton's laws of motion</a></span></li><li><span><a href="#Linear-momentum" data-toc-modified-id="Linear-momentum-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Linear momentum</a></span></li><li><span><a href="#Impulse" data-toc-modified-id="Impulse-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Impulse</a></span></li><li><span><a href="#Force" data-toc-modified-id="Force-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Force</a></span></li><li><span><a href="#Work" data-toc-modified-id="Work-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Work</a></span></li><li><span><a href="#Mechanical-energy" data-toc-modified-id="Mechanical-energy-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Mechanical energy</a></span><ul class="toc-item"><li><span><a href="#Kinetic-energy" data-toc-modified-id="Kinetic-energy-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Kinetic energy</a></span></li><li><span><a href="#Potential-energy" data-toc-modified-id="Potential-energy-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Potential energy</a></span></li><li><span><a href="#Power" data-toc-modified-id="Power-8.3"><span class="toc-item-num">8.3&nbsp;&nbsp;</span>Power</a></span></li></ul></li><li><span><a href="#Angular-momentum" data-toc-modified-id="Angular-momentum-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Angular momentum</a></span></li><li><span><a href="#Torque-(moment-of-force)" data-toc-modified-id="Torque-(moment-of-force)-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Torque (moment of force)</a></span></li><li><span><a href="#Mechanical-energy-for-angular-motion" data-toc-modified-id="Mechanical-energy-for-angular-motion-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Mechanical energy for angular motion</a></span><ul class="toc-item"><li><span><a href="#Kinetic-energy" data-toc-modified-id="Kinetic-energy-11.1"><span class="toc-item-num">11.1&nbsp;&nbsp;</span>Kinetic energy</a></span></li><li><span><a href="#Work" data-toc-modified-id="Work-11.2"><span class="toc-item-num">11.2&nbsp;&nbsp;</span>Work</a></span></li><li><span><a href="#Power" data-toc-modified-id="Power-11.3"><span class="toc-item-num">11.3&nbsp;&nbsp;</span>Power</a></span></li></ul></li><li><span><a href="#Principles-of-conservation" data-toc-modified-id="Principles-of-conservation-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Principles of conservation</a></span><ul class="toc-item"><li><span><a href="#Principle-of-conservation-of-linear-momentum" data-toc-modified-id="Principle-of-conservation-of-linear-momentum-12.1"><span class="toc-item-num">12.1&nbsp;&nbsp;</span>Principle of conservation of linear momentum</a></span></li><li><span><a href="#Principle-of-conservation-of-angular-momentum" data-toc-modified-id="Principle-of-conservation-of-angular-momentum-12.2"><span class="toc-item-num">12.2&nbsp;&nbsp;</span>Principle of conservation of angular momentum</a></span></li><li><span><a href="#Principle-of-conservation-of-mechanical-energy" data-toc-modified-id="Principle-of-conservation-of-mechanical-energy-12.3"><span class="toc-item-num">12.3&nbsp;&nbsp;</span>Principle of conservation of mechanical energy</a></span><ul class="toc-item"><li><span><a href="#Conservative-forces" data-toc-modified-id="Conservative-forces-12.3.1"><span class="toc-item-num">12.3.1&nbsp;&nbsp;</span>Conservative forces</a></span></li></ul></li></ul></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-internet" data-toc-modified-id="Video-lectures-on-the-internet-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Video lectures on the internet</a></span></li><li><span><a href="#References" data-toc-modified-id="References-15"><span class="toc-item-num">15&nbsp;&nbsp;</span>References</a></span></li></ul></div>
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
        ## The development of the laws of motion of bodies  

        "The theoretical development of the laws of motion of bodies is a problem of such interest and importance that it has engaged the attention of all the most eminent mathematicians since the invention of dynamics as a mathematical science by Galileo, and especially since the wonderful extension which was given to that science by Newton."

        &#8212; Hamilton, 1834 (apud Taylor, 2005).  
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
        - **Second law**: The acceleration of an object is directly proportional to the net force acting on the object and inversely proportional to the mass of the object:$\vec{F} = m \vec{a}$.
        - **Third law**: Whenever an object exerts a force$\vec{F}_1$(action) on a second object, this second object simultaneously exerts a force$\vec{F}_2$on the first object with the same magnitude but opposite direction (reaction):$\vec{F}_2 = −\vec{F}_1.$These three statements are astonishing in their simplicity and how much of knowledge they empower.   
        Isaac Newton was born in 1943 and his works that resulted in these equations and other discoveries were mostly done in the years of 1666 and 1667, when he was only 24 years old!  

        Here are these three laws in Newton's own words (from page 83 of Book I in the first American edition of the [*Philosophiæ Naturalis Principia Mathematica*](http://archive.org/details/newtonspmathema00newtrich):

        > LAW I.    
        > *Every body perseveres in its state of rest, or of uniform motion in a right line, unless it is compelled to change that state by forces impressed thereon.*   
        > LAW II.    
        > *The alteration of motion is ever proportional to the motive force impressed; and is made in the direction of the right line in which that force is impressed.*   
        > LAW III.   
        > *To every action there is always opposed an equal reaction: or the mutual actions of two bodies upon each other are always equal, and directed to contrary parts.*   

        And Newton carefully defined mass, motion, and force in the first page of the book I (page 73 of the [*Principia*](http://archive.org/details/newtonspmathema00newtrich)):  

        > DEFINITION I.   
        > *The quantity of matter is the measure of the same, arising from its density and bulk conjunctly.*   
        > ...It is this quantity that I mean hereafter everywhere under the name of body or mass.   
        > DEFINITION II.   
        > *The quantity of motion is the measure of the same, arising from the velocity and quantity of matter conjunctly.*    
        > The motion of the whole is the sum of the motions of all the parts; and therefore in a body double in quantity, with equal velocity, the motion is double; with twice the velocity, it is quadruple.   
        > DEFINITION IV.   
        > *An impressed force is an action exerted upon a body, in order to change its state, either of rest, or of moving uniformly forward in a right line.*
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Linear momentum

        From Definition II above, we can see that Newton defined as motion what we know today as linear momentum, the product between mass and velocity:$\vec{p} = m\vec{v}$So, in his second law, *alteration of motion is ever proportional to the motive force impressed*, if we understand that it was implicit that the *alteration* occurs in a certain time (or we can understand *force impressed* as force during a certain  time), Newton actually stated:$\vec{F} = \frac{\Delta\vec{p}}{\Delta t} \;\;\;\;\;\; \text{or}\;\;\;\;\;\; \vec{F}\Delta t = \Delta\vec{p}$What is equivalent to$\vec{F} = m\vec{a} \;$if mass is constant.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Impulse

        The mechanical linear impulse is a related concept and it can be derived from the second law of motion:$\vec{Imp} = \vec{F}\Delta t = m\Delta\vec{v}$And if the force varies with time:$\vec{Imp} = \sum_t \vec{F}(t)\Delta t$or using [infinitesimal calculus](http://en.wikipedia.org/wiki/Infinitesimal_calculus) (that it was independently developed by Newton himself and Leibniz):$\vec{Imp} = \int_t \vec{F}(t)dt$The concept of impulse due to a force that varies with time is often applied in biomechanics because it is common to measure forces (for example, with force plates) during human movement.  
        When such varying force is measured, the impulse can be calculated as the area under the force-versus-time curve:
        """
    )
    return


@app.cell
def _(np, plt):
    # simulate some data:
    t = np.arange(0, 1.01, 0.01)
    f = 1000*(-t**3+t**2)
    # plot:
    plt.rc('axes',  labelsize=16) 
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14)
    hfig, hax = plt.subplots(1,1, figsize=(8,5))
    hax.plot(t, f, linewidth=3)
    hax.set_xlim(-.1, 1.1)
    hax.grid()
    hax.set_ylabel('Force [N]')
    hax.set_xlabel('Time [s]')
    plt.fill(t, f, 'b', alpha=0.3)
    # area (impulse) with the trapz numerical integration method:
    from scipy.integrate import trapz
    imp = trapz(f, t)
    # plot a rectangle for the mean impulse value:
    plt.fill(np.array([t[0], t[0], t[-1], t[-1]]),
             np.array([0, imp, imp, 0]/(t[-1]-t[0])), 'r', alpha=0.3)
    s = '$i=F\Delta t = %.1f Ns$'%imp 
    plt.text(.4, 40, s, fontsize=18,
             bbox=dict(facecolor='white', edgecolor='white'));
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Force

        There are many manifestations of force we may experience during movement: gravitational, friction, ground reaction force, muscle force, buoyancy, elastic force, and other less visible such as electromagnetic, nuclear, etc. But in reality, all these different forces can be grouped in only four fundamental forces:

        - Strong force: hold the nucleus of an atom together. Range of action is$10^{-15}$m. 
        - Weak force: force acting between particles of the nucleus. Range of action is$10^{-18}$m.
        - Electromagnetic force: forces between electrical charges and the magnetic forces.
        - Gravity force: forces between masses; is the weakest of the four fundamental forces.

        In mechanics, forces can be classified as either contact or body forces. The contact force acts at the point of contact between two bodies. The body force acts on the entire body with no contact (e.g., gravity and electromagnetic forces).   
        In biomechanics, another useful classification is to divide the forces in either external or internal in relation to the human body. External forces result from interactions with an external body or environment (e.g., gravity and ground reaction forces). Internal forces result from interactions inside the body (e.g., the forces between bones).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Work

        The mechanical work of a force done on a body is the product between the component of the force in the direction of the resultant motion and the displacement:$W = \vec{F} \cdot \Delta\vec{x}$Where the symbol$\cdot$stands for the [scalar product](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/ScalarVector.ipynb) mathematical function.

        Mechanical work can also be understood as the amount of mechanical energy transferred into or out of a system.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Mechanical energy

        Mechanical energy is the sum of kinetic and potential energies.

        ### Kinetic energy$E_k = \frac{1}{2}mv^2$The linear momentum and the kinetic energy are related by:$\vec{p} = \frac{\partial E_k}{\partial\vec{v}}$### Potential energy

        The potential energy due to the gravitational force at the Earth's surface is:$E_p = mgh$The potential energy stored in a spring is:$E_p = \frac{1}{2}Kx^2$### Power$P = \frac{\Delta E}{\Delta t} \quad \text{and} \quad P = \vec{F} \cdot \vec{v}$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Angular momentum

        In analogy to the linear momentum, the angular momentum is the quantity of movement of a particle rotating around an axis at a distance$\vec{r}$:$\vec{L} = \vec{r} \times \vec{p}$For a particle rotating around an axis, the angular momentum can be expressed as:$\vec{L} = I \vec{\omega}$Where$I$is the rotational inertia or moment of inertia of the particle around the axis.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Torque (moment of force)

        In analogy to the second Newton's law for the linear case, torque or moment of force (or simply moment) is the time derivative of angular momentum:$\vec{M} = \frac{d\vec{L}}{dt} = \frac{d}{dt}(\vec{r} \times \vec{p}) = \frac{d\vec{r}}{dt} \times \vec{p} + \vec{r} \times \frac{d\vec{p}}{dt} = 0 + \vec{r} \times \vec{F}$$\vec{M} = \vec{r} \times \vec{F}$$\vec{M} = (r_x\:\mathbf{\hat{i}}+r_y\:\mathbf{\hat{j}}+r_z\:\mathbf{\hat{k}}) \times  (F_x\:\mathbf{\hat{i}}+F_y\:\mathbf{\hat{j}}+F_z\:\mathbf{\hat{k}})$Where the symbol$\times$stands for the [cross product](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/ScalarVector.ipynb) mathematical function.   
        The moment of force can be calculated as the determinant of the following matrix:$\vec{M} = \begin{bmatrix}
        \mathbf{\hat{i}} & \mathbf{\hat{j}} & \mathbf{\hat{k}} \\ 
        r_x & r_y & r_z \\
        F_x & F_y & F_z 
        \end{bmatrix}$$\vec{M} = (r_yF_z-r_zF_y)\mathbf{\hat{i}}+(r_zF_x-r_xF_z)\mathbf{\hat{j}}+(r_xF_y-r_yF_x)\mathbf{\hat{k}}$The moment of force can also be calculated by the geometric equivalent formula:$\vec{M} = \vec{r} \times \vec{F} = ||\vec{r}||\:||\vec{F}||\:sin(\theta)$Where$\theta$is the angle between the vectors$\vec{r}$and$\vec{F}$. 

        The animation below (from [Wikipedia](http://en.wikipedia.org/wiki/File:Torque_animation.gif)) illustrates the relationship between force ($\vec{F}$), torque ($\tau$), and momentum vectors ($\mathbf{p}$and$\vec{L}$):   

        <figure><img src="http://upload.wikimedia.org/wikipedia/commons/0/09/Torque_animation.gif" alt="Torque animation" width="300"/><figcaption><center><i>Figure. Relationship between force ($\mathbf{F}$), torque ($\tau$), and momentum vectors ($\mathbf{p}$and$\mathbf{L}$) (from [Wikipedia](http://en.wikipedia.org/wiki/File:Torque_animation.gif)).</i></center></figcaption></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Mechanical energy for angular motion
 
        ### Kinetic energy$E_k = \frac{1}{2}I\omega^2$### Work$W = \vec{M} \cdot \Delta\vec{\theta}$### Power$P = \frac{\Delta E}{\Delta t} \quad \text{and} \quad P = \vec{M} \cdot \vec{\omega}$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Principles of conservation

        ### Principle of conservation of linear momentum

        > *In a closed system with no external forces acting upon it, the total linear momentum of this system is constant.*

        ### Principle of conservation of angular momentum

        > *In a closed system with no external forces acting upon it, the total angular momentum of this system is constant.*

        ### Principle of conservation of mechanical energy

        > *In a closed system with no external forces acting upon it, the mechanical energy of this system is constant if only conservative forces act in this system.*


        #### Conservative forces

        A force is said to be conservative if this force produces the same work regardless of its trajectory between two points, if not the force is said to be non-conservative. Mathematically, the force$\vec{F}$is conservative if:$\oint \vec{F} \cdot d\vec{s} = 0$The gravitational force and the elastic force of an ideal spring are examples of conservative forces but friction force is not conservative. The forces generated by our muscles are also not conservative.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - Read chapter 0, What is mechanics, from Ruina and Rudra's book.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the internet

        - MIT OpenCourseWare: [History of Dynamics; Motion in Moving Reference Frames](https://youtu.be/GUvoVvXwoOQ)  
        - Fisica Universitária: [O que é a Mecânica](https://youtu.be/T10lXTek_JE)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Hibbeler (2010) Engineering Mechanics: Dynamics. 12th edition. (Hibbeler (2011) Dinâmica: Mecânica para Engenharia. 12a edição).
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
