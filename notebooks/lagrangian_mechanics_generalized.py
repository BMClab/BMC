import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Lagrangian mechanics in generalized coordinates

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
        The Lagrangian mechanics can be formulated completely independent of the Newtonian mechanics and Cartesian coordinates; Lagrange developed this new formalism based on the [principle of least action](http://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/principle_of_least_action.ipynb).  
        In this notebook, we will take a less noble path, we will deduce the Lagrange's equation from Newtonian mechanics.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Review-on-Newton's-laws-of-motion" data-toc-modified-id="Review-on-Newton's-laws-of-motion-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Review on Newton's laws of motion</a></span><ul class="toc-item"><li><span><a href="#Mechanical-energy" data-toc-modified-id="Mechanical-energy-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Mechanical energy</a></span></li></ul></li><li><span><a href="#Lagrange's-equation-in-Cartesian-Coordinates" data-toc-modified-id="Lagrange's-equation-in-Cartesian-Coordinates-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Lagrange's equation in Cartesian Coordinates</a></span></li><li><span><a href="#Generalized-coordinates" data-toc-modified-id="Generalized-coordinates-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Generalized coordinates</a></span></li><li><span><a href="#Lagrange's-equation" data-toc-modified-id="Lagrange's-equation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Lagrange's equation</a></span><ul class="toc-item"><li><span><a href="#Constraints" data-toc-modified-id="Constraints-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Constraints</a></span></li></ul></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-internet" data-toc-modified-id="Video-lectures-on-the-internet-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Video lectures on the internet</a></span></li><li><span><a href="#References" data-toc-modified-id="References-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Review on Newton's laws of motion

        The [Newton's laws of motion](https://en.wikipedia.org/wiki/Newton's_laws_of_motion) laid the foundation for classical mechanics. They describe the relationship between the motion of a body and the possible forces acting upon it.

        Consider the motion of a particle in three-dimensional space, its position in time can be represented by the following vector:
        <p>
        <span class="notranslate">$\vec{r}(t) = x(t)\hat{i} + y(t)\hat{j} + z(t)\hat{k}$</span>
    
        And given its position, the particle's velocity and acceleration are:
        <p>
        <span class="notranslate">$\begin{array}{l}
        \vec{v}(t) = \dfrac{\mathrm d \vec{r}(t)}{\mathrm d t} = \dfrac{d x(t)}{\mathrm d t}\hat{i} + \dfrac{d y(t)}{\mathrm d t}\hat{j} + \dfrac{d z(t)}{\mathrm d t}\hat{k} \\
        \vec{a}(t) = \dfrac{\mathrm d \vec{v}(t)}{\mathrm d t} = \dfrac{\mathrm d^2 \vec{r}(t)}{\mathrm d t^2} = \dfrac{d^2 x(t)}{\mathrm d t^2}\hat{i} + \dfrac{d^2 y(t)}{\mathrm d t^2}\hat{j} + \dfrac{d^2 z(t)}{\mathrm d t^2}\hat{k} 
        
        \end{array}$</span>

        The particle's linear momentum is defined as:
        <p>
        <span class="notranslate">$\vec{p}(t) = m\vec{v}(t)$</span>

        where$m$and$\vec{v}$are the mass and velocity of the body.

        Newton's second law relates the resultant force applied on the particle to the rate of variation of its linear momentum, and if the mass is constant:
        <p>
        <span class="notranslate">$\begin{array}{l}
        \vec{F}(t) = \dfrac{\mathrm d \vec{p}(t)}{\mathrm d t} = \dfrac{\mathrm d (m\vec{v}(t))}{\mathrm d t} \\
        \vec{F}(t) = m\vec{a}(t)    
        
        \end{array}$</span>

        From Newton's second law, if the position of the particle at any time is known, one can determine the resultant force acting on it. If the position is not known, but the resultant force is, the position of the particle can determined solving the following second order ordinary differential equation:
        <p>
        <span class="notranslate">$\frac{\mathrm d^2 \vec{r}(t)}{\mathrm d t^2} = \frac{\vec{F}(t)}{m}$</span>
    
        The differential equation above is referred as the equation of motion (EOM) of the particle. For example, a system of$N$particles will require$3N$EOMs to describe their motion.  
        The EOM has the general solution:
        <p>
        <span class="notranslate">$\vec{r}(t) = \int\!\bigg(\int\frac{\vec{F}(t)}{m} \, \mathrm{d}t\bigg) \, \mathrm{d}t$</span>
    
        which requires the determination of two constants, the initial position and velocity.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Mechanical energy

        A related physical quantity is the mechanical energy, which is the sum of kinetic and potential energies.  
        The kinetic energy,$T$of a particle is given by:
        <p>
        <span class="notranslate">$T = \frac{1}{2}m v^2$</span>
    
        Which can be expressed in terms of its linear momentum as:
        <p>
        <span class="notranslate">$T = \frac{1}{2m}  p^2$</span>
    
        And for a given coordinate of the particle's motion, its linear momentum can be obtained from its kinetic energy by:
        <p>
        <span class="notranslate">$\vec{p} = \frac{\partial T}{\partial \vec{v}}$</span>
    
        The potential energy$V$is the stored energy of a particle and its formulation is dependent on the force acting on the particle. For a conservative force dependent solely on the particle position, such as due to the gravitational field near the Earth surface or due to a linear spring, the force can be expressed in terms of the gradient of the potential energy:
        <p>
        <span class="notranslate">$\vec{F} = -\nabla V(\vec{r}) = -\frac{\partial V}{\partial x}\hat{i} - \frac{\partial V}{\partial y}\hat{j} - \frac{\partial V}{\partial z}\hat{k}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Lagrange's equation in Cartesian Coordinates

        For simplicity, let's first deduce the Lagrange's equation for a particle in Cartesian Coordinates and from Newton's second law.

        Because we want to deduce the laws of motion based on the mechanical energy of the particle, one can see that the time derivative of the expression for the linear momentum as a function of the kinetic energy, cf. Eq. (\ref{eq11}), is equal to the force acting on the particle and we can substitute the force in Newton's second law by this term:$\frac{\mathrm d }{\mathrm d t}\bigg(\frac{\partial T}{\partial \dot x}\bigg) = m\ddot x$We saw that a conservative force can also be expressed in terms of the potential energy of the particle, cf. Eq. (\ref{eq12}); substituting the right side of the equation above by this expression, we have:$\frac{\mathrm d }{\mathrm d t}\bigg(\frac{\partial T}{\partial \dot x}\bigg) = -\frac{\partial V}{\partial x}$Using the fact that:$\frac{\partial T}{\partial x} = 0 \quad and \quad \frac{\partial V}{\partial \dot x} = 0$We can write:$\frac{\mathrm d }{\mathrm d t}\bigg(\frac{\partial (T-V)}{\partial \dot x}\bigg) - \frac{\partial (T-V)}{\partial x} = 0$Defining the Lagrange or Lagrangian function,$\mathcal{L}$, as the difference between the kinetic and potential energy in the system:$\mathcal{L} = T - V$We have the Lagrange's equation in Cartesian Coordinates for a conservative force acting on a particle:$\frac{\mathrm d }{\mathrm d t}\bigg(\frac{\partial \mathcal{L}}{\partial \dot x}\bigg) - \frac{\partial \mathcal{L}}{\partial x} = 0$Once all derivatives of the Lagrangian function are calculated, this equation will be the equation of motion for the particle. If there are$N$independent particles in a three-dimensional space, there will be$3N$equations of motion for the system.  
        The set of equations above for a system are known as Euler–Lagrange's equations, or Lagrange's equations of the second kind.
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

        When describing a system of particles using any set of generalized coordinates,$q_1,\dotsc,q_{3N}$, these are related to, for example, the Cartesian coordinates by:$\begin{array}{rcl}
        q_i =q_i (x_1,\dotsc,x_{3N} ) \quad i=1,\dotsc,3N \\
        x_i =x_i (q_1,\dotsc,q_{3N} ) \quad i=1,\dotsc,3N 
        
        \end{array}$The Cartesian components of velocity as a function of generalized coordinates are:$\dot{x}_i =\frac{\mathrm d x_i (q_1, q_2,\dotsc,q_{3N} 
        )}{\mathrm d t}=\sum\limits_{j=1}^{3N} {\frac{\partial x_i }{\partial q_j }} 
        \frac{\mathrm d q_j }{\mathrm d t}$where for simplicity we omitted the explicit mention of the temporal dependence of each coordinate.

        That is, any Cartesian component of the particle velocity as a function of generalized coordinates is a function of all the components of position and velocity in the generalized coordinates:$\dot{x}_i = \dot{x}_i (q_1,\dotsc,q_{3N} ,\dot{q}_1,\dotsc,\dot{q}_{3N} ) \quad i=1,\dotsc,3N$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Lagrange's equation

        In analogy to Newtonian mechanics, one can think that the equations of motion can be obtained by equating the generalized force,$F_i$, to the temporal rate of change of each generalized momentum,$p_i$:$F_i =\frac{\partial p_i }{\partial t}$In the formula above, let's substitute the quantity$p_i$by its definition in terms of the kinetic energy:$\frac{\partial p_i }{\partial t} =\frac{\partial }{\partial t}\left( {\frac{\partial T}{\partial 
        \dot{q}_i }} \right)=\frac{\partial }{\partial t}\left( 
        {\sum\limits_{j=1}^{3N} {m_j \dot{x}_j \frac{\partial \dot{x}_j 
        }{\partial \dot{q}_i }} } \right)$where we used:$\frac{\partial T}{\partial \dot{q}_i }=\sum\limits_{j=1}^{3N} 
        {\frac{\partial T}{\partial \dot{x}_j }\frac{\partial \dot{x}_j 
        }{\partial \dot{q}_i }}$Using the [product rule](https://en.wikipedia.org/wiki/Product_rule), the derivative of the product in Eq. (\ref{eq29}) is:$\frac{\partial p_i }{\partial t}=\sum\limits_{j=1}^{3N} {m_j 
        \ddot{x}_j \frac{\partial \dot{x}_j }{\partial \dot{q}_i }} 
        +\sum\limits_{j=1}^{3N} {m_j \dot{x}_j \frac{\mathrm d }{\mathrm d t}\left( 
        {\frac{\partial \dot{x}_j }{\partial \dot{q}_i }} \right)}$But:$\frac{\partial \dot{x}_i }{\partial \dot{q}_j }=\frac{\partial x_i 
        }{\partial q_j } \quad because \quad \frac{\partial 
        \dot{x}_i }{\partial \dot{q}_j }=\frac{\partial x_i }{\partial 
        t}\frac{\partial t}{\partial q_j }=\frac{\partial x_i }{\partial q_j}$Then:$\frac{\partial p_i }{\partial t}=\sum\limits_{j=1}^{3N} {m_j 
        \ddot{x}_j \frac{\partial x_j }{\partial q_i }} 
        +\sum\limits_{j=1}^{3N} {m_j \dot{x}_j \frac{\mathrm d }{\mathrm d t}\left( 
        {\frac{\partial x_j }{\partial q_i }} \right)}$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The first term on the right side of the equation above is proportional to$m_j 
        \ddot{x}_j$and we will define as the generalized force,$Q_i$. But, different from Newtonian mechanics, the temporal variation of the generalized momentum is equal to the generalized force plus another term, which will investigate now. The last part of this second term can be derived as:$\frac{\mathrm d }{\mathrm d t}\left( {\frac{\partial x_j }{\partial q_i }} \right) = 
        \sum\limits_{k=1}^{3N} {\frac{\mathrm d }{\mathrm d q_k }\left( {\frac{\partial 
        x_j }{\partial q_i }} \right)\frac{\mathrm d q_k }{\mathrm d t}} =\sum\limits_{k=1}^{3N} 
        {\frac{\partial^2 x_j }{\partial q_k \partial q_i }\dot{q}_k }$where we used the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) for the differentiation:$\frac{\mathrm d }{\mathrm d t}\Big( {f\big({g(t)}\big)}\Big) = \frac{\partial f}{\partial g}\frac{\partial g}{\partial t}$But if we look at Eq. (\ref{eq_xdotqdot}) we see that the term at the right side of the Eq. (\ref{eq34}) can be obtained by:$\frac{\partial \dot{x}_j }{\partial q_i } = \frac{\partial }{\partial q_i }\left(\sum\limits_{k=1}^{3N} \frac{\partial 
        x_j }{\partial q_i }\dot{q}_k \right) = \sum\limits_{k=1}^{3N} 
        {\frac{\partial^2 x_j }{\partial q_k \partial q_i }\dot{q}_k }$Comparing the Eq. (\ref{eq34}) and Eq. (\ref{eq36}) we have:$\frac{\mathrm d }{\mathrm d t}\left( {\frac{\partial x_j }{\partial q_i }} \right) = 
        \frac{\mathrm d }{\mathrm d q_i}\left( {\frac{\partial x_j }{\partial t }} \right)$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        On the other hand, it is possible to relate the term$\partial \dot{x}_j / \partial q_i$to the derivative of kinetic energy with respect to the coordinate$q_i$:$\frac{\partial T}{\partial q_i }=\frac{\partial }{\partial q_i }\left( 
        {\sum\limits_{j=1}^{3N} {\frac{1}{2}m_j \dot{x}_j^2} } 
        \right)=\sum\limits_{j=1}^{3N} {m_j \dot{x}_j } \frac{\partial 
        \dot{x}_j }{\partial q_i }$where once again we used the chain rule for the differentiation.

        Using Eq. (\ref{eq_dotdxdq}), Eq. (\ref{eq38}) becomes$\frac{\partial T}{\partial q_i }=\sum\limits_{j=1}^{3N} {m_j 
        \dot{x}_j } \frac{\mathrm d }{\mathrm d t}\left( {\frac{\partial x_j }{\partial q_i }} 
        \right)$Returning to Eq. (\ref{eq33}), it can be rewritten as:$\frac{\mathrm d }{\mathrm d t}\left( {\frac{\partial T}{\partial \dot{q}_i }} \right) = Q_i + \frac{\partial T}{\partial q_i }$and$\frac{\mathrm d }{\mathrm d t}\left( {\frac{\partial T}{\partial \dot{q}_i }} \right) - \frac{\partial T}{\partial q_i } = Q_i$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's look at$Q_i$, the generalized force. It can be decomposed into two terms: 

        The first term, composed of the conservative forces, i.e. forces that can be written as potential gradients:$Q_C =-\frac{\partial V}{\partial q_i } \quad , \quad V=V\left( {q_1,\dotsc,q_{3N} } \right)$An example of conservative force is the gravitational force.

        And the second term, encompassing all non-conservative forces,$Q_{NC}$.  

        Then:$Q_i =-\frac{\partial V}{\partial q_i }+Q_{NCi} \quad , \quad V=V\left( {q_1,\dotsc,q_{3N} } \right)$The Eq. (\ref{eq41}) becomes$\frac{\mathrm d }{\mathrm d t}\left( {\frac{\partial T}{\partial \dot{q}_i }} 
        \right)-\frac{\partial T}{\partial q_i }=-\frac{\partial V}{\partial q_i} + Q_{NCi}$Rearranging, we have:$\frac{\mathrm d }{\mathrm d t}\left( {\frac{\partial \left( {T-V} \right)}{\partial 
        \dot{q}_i }} \right)-\frac{\partial \left( {T-V} \right)}{\partial q_i} = Q_{NCi}$This is possible because:$\frac{\partial V}{\partial \dot{q}_i} = 0$Defining:$\mathcal{L} \equiv \mathcal{L}(q_1,\dotsc,q_{3N} ,\dot{q}_1,\dotsc,\dot{q}_{3N} ) = T - V$as the Lagrange or Lagrangian function, we have the Lagrange's equation:$\frac{\mathrm d }{\mathrm d t}\left( {\frac{\partial \mathcal{L}}{\partial \dot{q}_i }} 
        \right)-\frac{\partial \mathcal{L}}{\partial q_i } = Q_{NCi} \quad i=1,\dotsc,3N$Once all derivatives of the Lagrangian function are calculated, this equation will be the equation of motion for each particle. If there are$N$independent particles in a three-dimensional space, there will be$3N$equations for the system.

        The set of equations above for a system are known as Euler–Lagrange equations, or Lagrange's equations of the second kind.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Constraints
 
        An important class of problems in mechanics, in which the Lagrangian equations are particularly useful, are composed of constrained systems. A constraint is a restriction on the freedom of movement of a particle or a system of particles (a constraint decreases the number of degrees of freedom of a system). A rigid body, or the movement of a pendulum, are examples of constrained systems. It can be shown, in a similar way, that the Lagrange equation, deduced here for a system of free particles, is also valid for a system of particles under the action of constraints.  
        The Euler-Lagrange equation, for a system of$3N$particles and with$k$constraints, is then defined as:$\frac{\mathrm d }{\mathrm d t}\left( {\frac{\partial \mathcal{L}}{\partial \dot{q}_i}} \right)-\frac{\partial \mathcal{L}}{\partial q_i } = Q_{NCi} \quad i=1,\dotsc,3N-k$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

        - [The Principle of Least Action in ](https://www.feynmanlectures.caltech.edu/II_19.html)  
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
        ## References

        - Goldstein H (1980) [Classical Mechanics](https://books.google.com.br/books?id=tJCuQgAACAAJ), 3rd ed., Addison-Wesley.  
        - Marion JB (1970) [Classical Dynamics of particles and systems](https://books.google.com.br/books?id=Ss43BQAAQBAJ), 2nd ed., Academic Press.  
        - Synge JL (1949) [Principles of Mechanics](https://books.google.com.br/books?id=qsYfENCRG5QC), 2nd ed., McGraw-hill.  
        - Taylor J (2005) [Classical Mechanics](https://archive.org/details/JohnTaylorClassicalMechanics). University Science Books. 
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
