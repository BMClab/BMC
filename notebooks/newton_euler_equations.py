import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Newton-Euler equations for rigid bodies

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
        <h1>Table of Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Mechanics" data-toc-modified-id="Mechanics-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Mechanics</a></span></li><li><span><a href="#Recapitulation" data-toc-modified-id="Recapitulation-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Recapitulation</a></span><ul class="toc-item"><li><span><a href="#Newton's-laws-of-motion" data-toc-modified-id="Newton's-laws-of-motion-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Newton's laws of motion</a></span></li><li><span><a href="#Linear-momentum" data-toc-modified-id="Linear-momentum-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Linear momentum</a></span></li><li><span><a href="#Angular-momentum" data-toc-modified-id="Angular-momentum-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Angular momentum</a></span></li><li><span><a href="#Torque-(moment-of-force)" data-toc-modified-id="Torque-(moment-of-force)-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Torque (moment of force)</a></span></li><li><span><a href="#Moment-of-inertia" data-toc-modified-id="Moment-of-inertia-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Moment of inertia</a></span></li></ul></li><li><span><a href="#Principle-of-transmissibility-and-Principle-of-moments" data-toc-modified-id="Principle-of-transmissibility-and-Principle-of-moments-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Principle of transmissibility and Principle of moments</a></span><ul class="toc-item"><li><span><a href="#Principle-of-transmissibility" data-toc-modified-id="Principle-of-transmissibility-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Principle of transmissibility</a></span></li><li><span><a href="#Varignon's-Theorem-(Principle-of-Moments)" data-toc-modified-id="Varignon's-Theorem-(Principle-of-Moments)-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Varignon's Theorem (Principle of Moments)</a></span></li></ul></li><li><span><a href="#Equivalent-systems" data-toc-modified-id="Equivalent-systems-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Equivalent systems</a></span></li><li><span><a href="#Mechanics-(dynamics)-of-rigid-bodies" data-toc-modified-id="Mechanics-(dynamics)-of-rigid-bodies-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Mechanics (dynamics) of rigid bodies</a></span></li><li><span><a href="#Euler's-laws-of-motion-(for-a-rigid-body)" data-toc-modified-id="Euler's-laws-of-motion-(for-a-rigid-body)-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Euler's laws of motion (for a rigid body)</a></span><ul class="toc-item"><li><span><a href="#Derivation-of-the-Euler's-laws-of-motion" data-toc-modified-id="Derivation-of-the-Euler's-laws-of-motion-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Derivation of the Euler's laws of motion</a></span></li></ul></li><li><span><a href="#Problems" data-toc-modified-id="Problems-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Mechanics

        In Mechanics we are interested in the study of motion (including deformation) and forces (and the relation between them) of anything in nature.  

        As a good rule of thumb, we model the phenomenon of interest as simple as possible, with just enough complexity to understand the phenomenon. 

        For example, we could model a person jumping as a particle (the center of gravity, with no size) moving in one direction (the vertical) if all we want is to estimate the jump height and relate that to the external forces to the human body. So, mechanics of a particle might be all we need. 

        However, if the person jumps and performs a somersault, to understand this last part of the motion we have to model the human body as one of more objects which displaces and rotates in two or three dimensions. In this case, we would need what is called mechanics of rigid bodies.

        If, besides the gross motions of the segments of the body, we are interested in understanding the deformation in the the human body segments and tissues, now we would have to describe the mechanical behavior of the body (e.g., how it deforms) under the action of forces. In this case we would have to include some constitutive laws describing the mechanical properties of the body.

        In the chapter mechanics of rigid bodies, the body deformation is neglected, i.e., the distance between every pair of points in the body is considered constant. Consequently, the position and orientation of a rigid body can be completely described by a corresponding coordinate system attached to it.

        Let's review some Newton's laws of motion for a particle and then extend these equations to motion of rigid bodies.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Recapitulation

        ### Newton's laws of motion

        The Newton's laws of motion describe the relationship between the forces acting on a body and the resultant linear motion due to those forces:

        - **First law**: An object will remain at rest or in uniform motion in a straight line unless an external force acts on the body.
        - **Second law**: The acceleration of an object is directly proportional to the net force acting on the object and inversely proportional to the mass of the object: <span class="notranslate">$\mathbf{\vec{F}} = m\mathbf{\vec{a}}.$</span>
        - **Third law**: Whenever an object exerts a force <span class="notranslate">$\mathbf{\vec{F}}_1$</span> (action) on a second object, this second object simultaneously exerts a force <span class="notranslate">$\mathbf{\vec{F}}_2$</span> on the first object with the same magnitude but opposite direction (reaction): <span class="notranslate">$\mathbf{\vec{F}}_2 = −\mathbf{\vec{F}}_1.$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Linear momentum

        The linear momentum, or quantity of motion, is defined as the product between mass and velocity:
        <span class="notranslate">$\mathbf{\vec{L}} = m\mathbf{\vec{v}}$</span>

        ### Angular momentum

        In analogy to the linear momentum, the angular momentum is the quantity of movement of a particle rotating around an axis passing through any point O at a distance$\mathbf{\vec{r}}$to the particle:

        <span class="notranslate">$\mathbf{\vec{H_O}} = \mathbf{\vec{r_{O}}} \times \mathbf{\vec{L}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Torque (moment of force)

        In analogy to the second Newton's law for the linear case, torque or moment of force (or simply moment) is the time derivative of angular momentum:

        <span class="notranslate">$\mathbf{\vec{M_O}} = \frac{d\mathbf{\vec{H_O}}}{dt} = \frac{d}{dt}(\mathbf{\mathbf{\vec{r}} \times \mathbf{\vec{L}}}) = \frac{d\mathbf{\vec{r_O}}}{dt} \times \mathbf{\vec{L}} + \mathbf{\vec{r_O}} \times \frac{d\mathbf{\vec{L}}}{dt} = \frac{d\mathbf{\vec{r_O}}}{dt} \times (m\mathbf{\mathbf{\vec{v}}}) + \mathbf{\vec{r_O}} \times \frac{d(m\mathbf{\vec{v}})}{dt} = \mathbf{\vec{v}} \times (m\mathbf{\mathbf{\vec{v}}}) + \mathbf{\vec{r_O}} \times \frac{d(m\mathbf{\vec{v}})}{dt} = 0 + \mathbf{\vec{r_O}} \times \mathbf{\vec{F}}$$\mathbf{\vec{M_O}} = \mathbf{\vec{r_O}} \times \mathbf{\vec{F}}$$\mathbf{\vec{M_O}} = (r_{O_x}\:\mathbf{\hat{i}}+r_{O_y}\:\mathbf{\hat{j}}+r_{O_z}\:\mathbf{\hat{k}}) \times  (F_x\:\mathbf{\hat{i}}+F_y\:\mathbf{\hat{j}}+F_z\:\mathbf{\hat{k}})$</span>

        Where the symbol$\times$stands for the [cross product](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/ScalarVector.ipynb) mathematical function.   
        The the moment of force can be calculated as the determinant of the following matrix:

        <span class="notranslate">$\mathbf{\vec{M_O}} = \det \begin{bmatrix}
        \mathbf{\hat{i}} & \mathbf{\hat{j}} & \mathbf{\hat{k}} \\ 
        r_{O_x} & r_{O_y} & r_{O_z} \\
        F_x & F_y & F_z 
        \end{bmatrix}$</span>

        <span class="notranslate">$\mathbf{\vec{M_O}} = (r_{O_y}F_z-r_{O_z}F_y)\mathbf{\hat{i}}+(r_{O_z}F_x-r_{O_x}F_z)\mathbf{\hat{j}}+(r_{O_x}F_y-r_{O_y}F_x)\mathbf{\hat{k}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The magnitude of moment of force can also be calculated by the geometric equivalent formula:

        <span class="notranslate">$||\mathbf{\vec{M_O}}|| = ||\mathbf{r_O} \times \mathbf{\vec{F}}|| = ||\mathbf{\vec{r_O}}||\:||\mathbf{\vec{F}}||\sin(\theta)$</span>

        Where <span class="notranslate">$\theta$</span> is the angle between the vectors <span class="notranslate">$\mathbf{\vec{r_O}}$</span> and <span class="notranslate">$\mathbf{\vec{F}}$<span>. 

        The animation below illustrates the relationship between force, torque, and momentum vectors:   

        <center><figure><img src="../images/TorqueAnim.gif" alt="Torque animation" width="300"/><figcaption><i>Figure. Relationship between force ($\mathbf{\vec{F}}$), torque ($\mathbf{\vec{M}}$), linear momentum ($\mathbf{\vec{L}}$) and angular momentum ($\mathbf{\vec{H}}$). Adapted from <a href="https://en.wikipedia.org/wiki/File:Torque_animation.gif">Wikipedia</a>.</i></figcaption></figure></center>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Moment of inertia

        Let's use the example above, where <span class="notranslate">$\mathbf{\vec{r_O}}$</span> and <span class="notranslate">$\mathbf{\vec{F}}$</span> are orthogonal and derive an expression for the magnitude of these quantities as the equivalent of Newton's second law for angular motion:

        <span class="notranslate">$M_O = r_OF = r_Oma$</span>

        Replacing the linear acceleration$a$by the angular acceleration$\alpha$:

        <span class="notranslate">$M_O = r_Omr_O\alpha = mr_O^2 \alpha$</span>

        In analogy to Newton's second law, where the constant of proportionality between$a$and$F$is called inertia (mass), the constant of proportionality between$M_O$and$\alpha$is called rotational inertia or moment of inertia,$I_O=mr_O^2$for a particle with mass$m$rotating at a distance$r$from the center of rotation O.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Principle of transmissibility and Principle of moments

        On the effects of forces, there are two important principles:

        ### Principle of transmissibility

        > *For rigid bodies with no deformation, an external force can be applied at any point on its line of action without changing the resultant effect of the force.*

        ### Varignon's Theorem (Principle of Moments)

        > *The moment of a force about a point is equal to the sum of moments of the components of the force about the same point.*   
        Note that the components of the force don't need to be orthogonal.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Equivalent systems


        A set of forces and moments is considered equivalent if its resultant force and sum of the moments computed relative to a given point are the same. Normally, we want to reduce all the forces and moments being applied to a body into a single force and a single moment.

        We have done this with particles for the resultant force. The resultant force is simply the sum of all the forces being applied to the body.

        <span class="notranslate">$\vec{\bf{F}} = \sum\limits_{i=1}^n \vec{\bf{F_i}}$</span>


        where <span class="notranslate">$\vec{\bf{F_i}}$</span> is each force applied to the body.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Similarly, the total moment applied to the body relative to a point O is:

        <span class="notranslate">$\vec{\bf{M_O}} = \sum\limits_{i}\vec{\bf{r_{i/O}}} \times \vec{\bf{F_i}} + \sum\limits_{i}\vec{\bf{M_{F_i}}}$</span>

        where <span class="notranslate">$\vec{\bf{r_{i/O}}}$</span> is the vector from the point O to the point where the force <span class="notranslate">$\vec{\bf{F_i}}$</span> is being applied and$\vec{\bf{M_{F_i}}}$are the free moments. 

        Free moments are moments created by a pair of forces. They do not sum um to the resultant force, since they have have the same magnitude and opposite directions, but create a moment because there is a distance between these forces.

        <center><figure><img src="../images/freeMoment.png" width=500/></figure></center>

        But where the resultant force should be applied in the body? If the resultant force were applied to any point different from the point O, it would produce an additional  moment to the body relative to point O. So, the resultant force must be applied to the point O.

        So, any set of forces can be reduced to a moment relative to a chosen point O and a resultant force applied to the point O.

        To compute the resultant force and moment relative to another point O', the new moment is:

        <span class="notranslate">$\vec{\bf{M_{O'}}} = \vec{\bf{M_O}} + \vec{\bf{r_{O'/O}}} \times \vec{\bf{F}}$</span>

        And the resultant force is the same.

        It is worth to note that if the resultant force  <span class="notranslate">$\vec{\bf{F}}$</span> is zero, than the moment is the same relative to any point.

        <center><figure><img src="../images/equivalentSystem.png" width=850/></figure></center>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Mechanics (dynamics) of rigid bodies

        A [rigid body](https://en.wikipedia.org/wiki/Rigid_body) is a model (an idealization) for a body in which deformation is neglected, i.e., the distance between every pair of points in the body is considered constant. This definition also also implies that the total mass of a rigid body is constant.

        Consequently, the motion of a rigid body can be completely described by its pose (position and orientation) in space. In a three-dimensional space, at least three coordinates and three angles are necessary to describe the pose of the rigid body, totalizing six degrees of freedom for a rigid body. This also implies that we will need six equations of motion for these components to describe the dynamics of a rigid body.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Euler's laws of motion (for a rigid body)

        Euler's laws of motion extend Newton's laws of motion for particles for the motion of a rigid body.

        **First law**: The linear momentum of a body is equal to the product of the mass of the body and the velocity of its center of mass:   

        <span class="notranslate">$\mathbf{\vec{L}} = m\mathbf{\vec{v}}_{cm}$</span>

        And calculating the time derivative of this equation:

        <span class="notranslate">$\mathbf{\vec{F}} = m\mathbf{\vec{a}}_{cm}$</span>

        **Second law**: The rate of change of angular momentum about a point that is fixed in an inertial reference frame is equal to the resultant external moment of force about that point:

        <span class="notranslate">$\mathbf{\vec{M_O}} = \frac{d\mathbf{\vec{H_O}}}{dt}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Derivation of the Euler's laws of motion

        **First law**: 

        The sum of the linear momentum of all the particles of a rigid body (considering the body as a discrete sum of elements, but this also holds for the continuous case):

        <span class="notranslate">$\mathbf{\vec{L}} = \sum m_i\mathbf{\vec{v}}_i$</span>

        Looking at the definition of center of mass:

        <span class="notranslate">$\mathbf{\vec{r}}_{cm} = \frac{1}{m_{B}}\sum m_{i}\mathbf{\vec{r}}_i \quad \text{where} \quad m_{B} = \sum m_{i}$<span>

        By differentiation, the velocity of the center of mass is:

        <span class="notranslate">$\mathbf{\vec{v}}_{cm} = \frac{1}{m_{B}}\sum m_{i}\mathbf{\vec{v}}_i$</span>

        And finally:

        <span class="notranslate">$\mathbf{\vec{L}} = m_{B} \mathbf{\vec{v}}_{cm} = m_B \mathbf{\vec{v}}_{cm}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can get the second equation of the first law calculating the time derivative of the equation above.  
        Another way to derive this second equation is considering the effects of all forces acting on each particle of the rigid body and apply Newton's second law to them:

        <span class="notranslate">$\sum \mathbf{\vec{F}}_i = \sum m_i\mathbf{\vec{a}}_i$</span>
    
        With respect to the origin of these forces, they can be divided in two types: external and internal forces to the rigid body. Internal forces are interaction forces between particles inside the body and because of Newton's third law (action and reaction) they cancel each other. So, the equation above becomes:

        <span class="notranslate">$\sum \mathbf{\vec{F}}_{external} = \sum m_i\mathbf{\vec{a}}_i$</span>
    
        But the acceleration of the center of mass is:

        <span class="notranslate">$\mathbf{\vec{a}}_{cm} = \frac{1}{m_B}\sum m_{i}\mathbf{\vec{a}}_i$</span>

        And finally:

        <span class="notranslate">$\mathbf{\vec{F}} = \sum \mathbf{\vec{F}}_{external} = m_B\mathbf{\vec{a}}_{cm}$</span>

        This means that for a rigid body the internal forces between the particles of the body do not contribute to changing the total momentum nor changing the acceleration of the center of mass. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Second law**: 

        For a complete derivation of the second Euler's law of motion, see for example Taylor (2005) or [http://emweb.unl.edu/NEGAHBAN/EM373/note19/note19.htm](http://emweb.unl.edu/NEGAHBAN/EM373/note19/note19.htm).

        Let's derive the second Euler's law of motion for a simpler case of a rigid body rotating in a plane.  

        First, a general consideration about the total angular momentum of a rotting rigid body:  
        The total angular momentum of a rigid body rotating around a point$O$can be expressed as the angular momentum of the body center of mass around the point$O$plus the sum of the angular momentum of each particle around the body center of mass (for a proof see page 368 of Taylor, 2005):

        <span class="notranslate">$\mathbf{\vec{H_O}} = \mathbf{\vec{r}}_{cm/O} \times m\mathbf{\vec{v}}_{cm/O} + \sum \mathbf{\vec{r}}_{i/cm} \times m_i\mathbf{\vec{v}}_{i/cm}$</span>

        For a two-dimensional case, where the rigid body rotates around its own center of mass and also rotates around another parallel axis (fixed), the second term of the right side of the equation above can be simplified to <span class="notranslate">$\sum (m_i\mathbf{r}^2_{i/cm}) \mathbf{\vec{\omega}}$</span> and calculating the time derivative of the whole equation,  

        <span class="notranslate">$\mathbf{\vec{M_O}} = \frac{d\mathbf{\vec{H_O}}}{dt} \rightarrow \mathbf{\vec{M_O}} = \frac{d\left( \mathbf{\vec{r}}_{cm/O} \times m\mathbf{\vec{v}}_{cm/O} + \sum \mathbf{\vec{r}}_{i/cm} \times m_i\mathbf{\vec{v}}_{i/cm} \right)}{dt}$</span>

        the second Euler's law of motion simplifies to: 

        <span class="notranslate">$\mathbf{\vec{M_O}} = \mathbf{\vec{r}}_{cm/O} \times m\mathbf{\vec{a}}_{cm} + I_{cm} \mathbf{\vec{\alpha}}$</span>

        where <span class="notranslate">$\mathbf{\vec{r}}_{cm}$</span> is the position vector of the center of mass with respect to the point$O$about which moments are summed, <span class="notranslate">$\mathbf{\vec{\alpha}}$</span> is the angular acceleration of the body about its center of mass, and <span class="notranslate">$I_{cm}$</span> is the moment of inertia of the body about its center of mass.

        If <span class="notranslate">$d$</span> is the (shortest) distance between the point$O$and the line of the acceleration vector, then the equation above becomes:

        <span class="notranslate">$\mathbf{M} = ma_{cm}d + I_{cm} \mathbf{\alpha}$</span>
    
        Note that if the acceleration of the center of mass is zero or the sum of moments of force is calculated around the center of mass (then$\mathbf{r}_{cm}=0$), this case of rotation in a plane simplifies to the well-known simple equation:

        <span class="notranslate">$\mathbf{\vec{M_{cm}}} = I_{cm} \mathbf{\vec{\alpha}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        *Three-dimensional case*

        In the three-dimensional space, if we describe the rotation of a rigid body using a rotating reference frame with axes parallel to the principal axes of inertia (referred by the subscripts 1,2,3) of the body, the Euler's second law becomes:   

        <span class="notranslate">$M_1 = I_1\dot{\omega_1} + (I_3-I_2)\omega_2\omega_3$$M_2 = I_2\dot{\omega_2} + (I_1-I_3)\omega_3\omega_1$$M_3 = I_3\dot{\omega_3} + (I_2-I_1)\omega_1\omega_2$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. (Sample problem 2/5, Taylor's book)Calculate the magnitude of the moment about the base point *O* of the 600-N force in five different ways for the structure shown below (hint: use the equation for torque in different ways, and also the principles of moments and of transmissibility)[.](../images/torque.png)  
        <center><figure><img src="https://github.com/BMClab/BMC/blob/master/images/torque2.png?raw=1" alt="Torque" width="250"/></figure></center>

        2. (Example 17.7, Hibbeler's book) A uniform$50kg$crate rests on a horizontal surface for which the coefficient of kinetic friction is$\mu_k = 0.2$. Determine the acceleration if a force of$P = 600 N$is applied to the crate as shown in the figure.  
        <center><figure><img src="../images/hibbeler_17_7.png" alt="crate" width="350"/></figure></center>  

        3. (Example 17.10, Hibbeler's book) At the instant shown in the figure, the$20kg$slender rod has an angular velocity of$\omega = S rad/s$. Determine the angular acceleration and the horizontal and vertical components of reaction of the pin on the rod at this instant.  
        <center><figure><img src="../images/hibbeler_17_10.png" alt="rod" width="350"/></figure></center>  

        4. (Example 17.12, Hibbeler's book) The slender rod shown in the figure has a mass$m$and length$l$and is released from rest when$\theta = 0^o$. Determine the horizontal and vertical components of force which the pin at$A$exerts on the rod at the instant$\theta = 90^o$.  
        <center><figure><img src="../images/hibbeler_17_12.png" alt="pendulum" width="300"/></figure></center>  

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
