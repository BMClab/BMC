import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Multibody dynamics of one and two-link systems

        > Marcos Duarte  
        > Laboratory of Biomechanics and Motor Control ([http://demotu.org/](http://demotu.org/))  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The human body is composed of multiple interconnected segments (which can be modeled as rigid or flexible) and each segment may have translational and rotational movement. The part of mechanics for the study of movement and forces of interconnected bodies is called [multibody system](http://en.wikipedia.org/wiki/Multibody_system) or multibody dynamics. 

        There are different approaches to deduce the kinematics and dynamics of such bodies, the most common are the [Newton-Euler](http://en.wikipedia.org/wiki/Newton%E2%80%93Euler_equations) and the [Langrangian](http://en.wikipedia.org/wiki/Lagrangian_mechanics) formalisms. The Newton-Euler formalism is based on the well known Newton-Euler equations. The Langrangian formalism uses the [principle of least action](http://en.wikipedia.org/wiki/Principle_of_least_action) and describes the movement based on [generalized coordinates](http://en.wikipedia.org/wiki/Generalized_coordinates), a set of parameters (typically, a convenient minimal set) to describe the configuration of the system taking into account its constraints. For a system with multiple bodies and several constraints, e.g., the human body, it is easier to describe the dynamics of such system using the Langrangian formalism. 
 
        Zajac and Gordon (1989) and Zajac (1993) offer excellent discussions about applying multibody system concepts to understanding human body movement.

        Next, we will study two simple problems of multibody systems in the context of biomechanics which we can handle well using the Newton-Euler approach.  
        First a planar one-link system (which is not a multibody), which can represent the movement of one limb of the body or the entire body as a single inverted pendulum.  
        Second, a planar two-link system, which can represent the movement of two segments of the body, e.g., upper arm and forearm. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Newton-Euler equations

        For a two-dimensional movement in the$XY$plane, the Newton-Euler equations are:  

        \begin{align}
        \left\{ \begin{array}{l l}
        \sum F_X &=& m \ddot{x}_{cm} \\
        \\
        \sum F_Y &=& m \ddot{y}_{cm} \\
        \\
        \sum M_Z &=& I_{cm} \ddot{\alpha}_Z
        \end{array} \right.
        \label{}
        \end{align}

        Where the movement is described around the body center of mass ($cm$).$(F_X,\,F_Y)$and$M_Z$are, respectively, the forces and moment of forces (torques) acting on the body.$(\ddot{x}_{cm},\,\ddot{y}_{cm})$and$\ddot{\alpha}_Z$are, respectively, the linear and angular accelerations.$I_{cm}$is the body moment of inertia around the body center of mass at the$Z$axis.  

        Let's use Sympy to derive some of the characteristics of the systems.
        """
    )
    return


@app.cell
def _():
    import sympy as sym
    from sympy import Symbol, symbols, cos, sin, Matrix, simplify
    from sympy.vector import CoordSys3D
    from sympy.physics.mechanics import dynamicsymbols, mlatex, init_vprinting
    init_vprinting()
    from IPython.display import display, Math

    eq = lambda lhs, rhs: display(Math(lhs + '=' + mlatex(rhs)))
    eq = lambda lhs, rhs: display(Math(r'\begin{array}{l l}' + lhs +
                                       '&=&' + mlatex(rhs) + r'\end{array}'))
    return (
        CoordSys3D,
        Math,
        Matrix,
        Symbol,
        cos,
        display,
        dynamicsymbols,
        eq,
        mlatex,
        simplify,
        sin,
        symbols,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## One-link system

        Let's study the dynamics of a planar inverted pendulum as a model for the movement of a human body segment with an external force acting on the segment (see Figure 1). 

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/invpend1.png?raw=1" alt="Inverted pendulum"/><figcaption><i><center>Figure 1. Planar inverted pendulum (one link attached to a fixed body by a hinge joint in a plane) with a joint actuators and corresponding free body diagram as a model of a human body segment. See text for notation convention.</center></i></figcaption>

        The following notation convention will be used for this problem:  

         -$L$is the length of the segment.  
         -$d$is the distance from the joint of the segment to its center of mass position.  
         -$m$is the mass of the segment. 
         -$g$is the gravitational acceleration (+).   
         -$\alpha$is the angular position of the joint w.r.t. horizontal,$\ddot{\alpha_i}$is the corresponding angular acceleration.  
         -$I$is the moment of inertia of the segment around its center of mass position.  
         -$F_{r}$is the joint reaction force.  
         -$F_{e}$is the external force acting on the segment.
         -$T$is the joint moment of force (torque). 
 
        In the case of a human body segment, muscles responsible for the movement of the segment are represented as a single pair of antagonistic joint actuators (e.g., flexors and extensors). We will consider that all joint torques are generated only by these muscles (we will disregard the torques generated by ligaments and other tissues) and the total or net joint torque will be the sum of the torques generated by the two muscles:$T \quad=\quad T_{net} \quad=\quad T_{extension} - T_{flexion}$Where we considered the extensor torque as positive (counter-clockwise). In what follows, we will determine only the net torque and we will be unable to decompose the net torque in its components.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Kinetics
 
        From the free body diagram, the Newton-Euler equations for the planar inverted pendulum are (writing the equation for the torques around the center of mass):$\begin{array}{l l}
        F_{r,x} + F_{e,x} & = & m\ddot{x} \\
        \\
        F_{r,y} - mg + F_{e,y} & = & m\ddot{y} \\
        \\
        T + dF_{r,x}\sin\alpha - dF_{r,y}\cos\alpha - (L-d)F_{e,x}\sin\alpha + (L-d)F_{e,y}\cos\alpha & = & I\ddot{\alpha}
        \end{array}$However, manually placing the terms in the Newton-Euler equations as we did above where we calculated the signs of the cross products is error prone. We can avoid this manual placing by treating the quantities as vectors and express them in matricial form:$\begin{array}{l l}
        \vec{\mathbf{F}}_r + \vec{\mathbf{F}}_g + \vec{\mathbf{F}}_e &=& m\ddot{\vec{\mathbf{r}}} \\
        \\
        \vec{\mathbf{T}} + \vec{\mathbf{r}}_{cm,j} \times \vec{\mathbf{F}}_r + \vec{\mathbf{r}}_{cm,e} \times \vec{\mathbf{F}}_e &=& I\ddot{\vec{\mathbf{\alpha}}}
        \end{array}$Where:$\begin{array}{l l}
        \begin{bmatrix} F_{rx} \\ F_{ry} \\ 0 \end{bmatrix}  + \begin{bmatrix} 0 \\ -g \\ 0 \end{bmatrix}  + \begin{bmatrix} F_{ex} \\ F_{ey} \\ 0 \end{bmatrix} &=& m\begin{bmatrix} \ddot{x} \\ \ddot{y} \\ 0 \end{bmatrix} , \quad \begin{bmatrix} \hat{i} \\ \hat{j} \\ \hat{k} \end{bmatrix}
        \\
        \begin{bmatrix} 0 \\ 0 \\ T_z \end{bmatrix}  + \begin{bmatrix} -d\cos\alpha \\ -d\sin\alpha \\ 0 \end{bmatrix}  \times \begin{bmatrix} F_{rx} \\ F_{ry} \\ 0 \end{bmatrix} + \begin{bmatrix} (L-d)\cos\alpha \\ (L-d)\sin\alpha \\ 0 \end{bmatrix}  \times \begin{bmatrix} F_{ex} \\ F_{ey} \\ 0 \end{bmatrix} &=& I_z\begin{bmatrix} 0 \\ 0 \\ \ddot{\alpha} \end{bmatrix} , \quad \begin{bmatrix} \hat{i} \\ \hat{j} \\ \hat{k} \end{bmatrix}
        \end{array}$Note that$\times$represents the cross product, not matrix multiplication. Then, both in symbolic or numeric manipulation we would use the cross product function to perform part of the calculations.  
        There are different computational tools that can be used for the formulation of the equations of motion. For instance, Sympy has a module, [Classical Mechanics](http://docs.sympy.org/dev/modules/physics/mechanics/), and see [this list](http://real.uwaterloo.ca/~mbody/#Software) for other software.  
        Let's continue with the explicit manual formulation of the equations for now.  

        We can rewrite the equation for the moments of force in a form that doesn't explicitly involve the joint reaction force expressing the moments of force around the joint center:$T - mgd\cos\alpha - LF_{e,x}\sin\alpha + LF_{e,y}\cos\alpha \quad=\quad I_o\ddot{\alpha}$Where$I_o$is the moment of inertia around the joint,$I_o=I_{cm}+md^2$, using the parallel axis theorem.  

        The torque due to the joint reaction force does not appear on this equation; this torque is null because by the definition the reaction force acts on the joint. If we want to determine the joint torque and we know the kinematics, we perform inverse dynamics:$T \quad=\quad I_o\ddot{\alpha} + mgd \cos \alpha + LF_{e,x}\sin\alpha - LF_{e,y}\cos\alpha$If we want to determine the kinematics and we know the joint torque, we perform direct dynamics:$\ddot{\alpha} \quad=\quad I_o^{-1}[T - mgd \cos \alpha - LF_{e,x}\sin\alpha + LF_{e,y}\cos\alpha ]$The expression above is a second-order differential equation which typically is solved numerically. So, unless we are explicitly interested in estimating the joint reaction forces, we don't need to use them for calculating the joint torque or simulate movement. Anyway, let's look at the kinematics of this problem to introduce some important concepts which will be needed later.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Kinematics

        A single planar inverted pendulum has one degree of freedom, the rotation movement of the segment around the pin joint. In this case, if the angular position$\alpha(t)$is known, the coordinates$x(t)$and$y(t)$of the center of mass and their derivatives can be readily determined (a process referred as [forward kinematics)](http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/KinematicChain.ipynb#Forward-and-inverse-kinematics):
        """
    )
    return


@app.cell
def _(Symbol, dynamicsymbols, symbols):
    t = Symbol('t')
    d, L = symbols('d L', positive=True)
    a = dynamicsymbols('alpha')
    return L, a, d, t


@app.cell
def _(a, cos, d, eq, sin, t):
    x, y = d*cos(a), d*sin(a)
    xd, yd = x.diff(t), y.diff(t)
    xdd, ydd = xd.diff(t), yd.diff(t)

    eq(r'x', x)
    eq(r'\dot{x}', xd)
    eq(r'\ddot{x}', xdd)
    eq(r'y', y)
    eq(r'\dot{y}', yd)
    eq(r'\ddot{y}', ydd)
    return x, xdd, y, ydd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The terms in$\ddot{x}$and$\ddot{y}$proportional to$\dot{\alpha}^2$are components of the centripetal acceleration on the body. As the name suggests, the [centripetal](http://en.wikipedia.org/wiki/Centripetal_force) acceleration is always directed to the center (towards the joint) when the segment is rotating. See the notebook [Kinematic chain](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/KinematicChain.ipynb) for more on that.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We could also use the methods of the Sympy physics/mechanics module and explicitly create a coordinate system in 3-D space, which will employ the versors$\hat{\mathbf{i}}, \hat{\mathbf{j}}, \hat{\mathbf{k}}$for representing the vector components:
        """
    )
    return


@app.cell
def _(CoordSys3D, a, cos, d, eq, sin, t):
    G = CoordSys3D('')
    r = d*cos(a)*G.i + d*sin(a)*G.j + 0*G.k
    rd = r.diff(t)
    rdd = r.diff(t, 2)

    eq(r'\vec{\mathbf{r}}', r)
    eq(r'\dot{\vec{\mathbf{r}}}', rd)
    eq(r'\ddot{\vec{\mathbf{r}}}', rdd)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        But for now, let's continue writing the components ourselves.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As an exercise, let's go back to the Newton-Euler equation for the sum of torques around the center of mass where the torques due to the joint reaction forces are explicit.  
        From the equation for the the sum of forces, hence we have expressions for the linear accelerations, we can isolate the reaction forces and substitute them on the equation for the torques. With a little help from Sympy:
        """
    )
    return


@app.cell
def _(a, symbols, t):
    m, I, g = symbols('m I g', positive=True)
    Fex, Fey = symbols('F_ex F_ey')
    add = a.diff(t, 2)
    return Fex, Fey, I, add, g, m


@app.cell
def _(Fex, Fey, eq, g, m, xdd, ydd):
    Frx = m*xdd - Fex
    Fry = m*ydd + m*g - Fey
    eq(r'F_{rx}', Frx)
    eq(r'F_{ry}', Fry)
    return Frx, Fry


@app.cell
def _(Fex, Fey, Frx, Fry, I, L, a, add, cos, d, eq, sin):
    T = I*add - d*sin(a)*Frx + d*cos(a)*Fry + (L-d)*sin(a)*Fex - (L-d)*cos(a)*Fey
    eq(r'T', T)
    return (T,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This equation for the torques around the center of mass of only one rotating segment seems too complicated. The equation we derived before for the torques around the joint was much simpler. However, if we look at the terms on this last equation, we can simplify most of them. Let's use Sympy to simplify this equation:
        """
    )
    return


@app.cell
def _(T, eq, simplify):
    T_1 = simplify(T)
    eq('T', T_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And we are back to the more simple equation we've seen before. The first two terms on the right side are the torque due to the external force, the third and fourth are the moment of inertia around the joint (use the theorem of parallel axis) times the acceleration, and the last term is the gravitational torque.  

        But what happened with all the other terms in the equation?  

        First, the terms proportional to the angular acceleration were just components from each direction of the 'inertial' torque that when summed resulted in$md^2\ddot{\alpha}$. 
        Second, the terms proportional to$\dot{\alpha}^2$are components of the torque due to the centripetal force (acceleration). But the centripetal force passes through the joint as well as through the center of mass, i.e., it has zero lever arm and this torque should be zero. Indeed, when summed these terms are canceled out.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The Jacobian matrix

        Another way to deduce the velocity and acceleration of a point at the rotating link is to use the [Jacobian matrix](http://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) (see [Kinematic chain](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/KinematicChain.ipynb)).  
        Remember that in the context of kinematic chains, the Jacobian relates changes in the joint space to changes in the Cartesian space. The Jacobian is a matrix of all first-order partial derivatives of the linear position vector of the endpoint with respect to the angular position vector.  
        For the center of mass of the planar one-link system, this means that the Jacobian matrix is:$\mathbf{J} \quad=\quad
        \begin{bmatrix}
        \dfrac{\partial x}{\partial \alpha} \\
        \dfrac{\partial y}{\partial \alpha} \\
        \end{bmatrix}$"""
    )
    return


@app.cell
def _(Matrix, a, eq, x, y):
    r_1 = Matrix([x, y])
    J = r_1.diff(a)
    eq('\\mathbf{J}', J)
    return (r_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And Sympy has a function to calculate the Jacobian:
        """
    )
    return


@app.cell
def _(a, eq, r_1):
    J_1 = r_1.jacobian([a])
    eq('\\mathbf{J}', J_1)
    return (J_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The linear velocity of a point in the link will be given by the product between the Jacobian of the kinematic link and its angular velocity:$\vec{\mathbf{v}} \quad=\quad \mathbf{J} \dot{\vec{\alpha}}$Using Sympy, the linear velocity of the center of mass is:
        """
    )
    return


@app.cell
def _(J_1, a, eq, t):
    vel = J_1 * a.diff(t)
    eq('\\begin{bmatrix} \\dot{x} \\\\ \\dot{y} \\end{bmatrix}', vel)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the linear acceleration of a point in the link will be given by the derivative of this last expression:$\vec{\mathbf{a}} \quad=\quad \dot{\mathbf{J}} \dot{\vec{\alpha}} + \mathbf{J} \ddot{\vec{\alpha}}$And using Sympy again, the linear acceleration of the center of mass is:
        """
    )
    return


@app.cell
def _(J_1, a, eq, t):
    acc = (J_1 * a.diff(t)).diff(t)
    eq('\\begin{bmatrix} \\ddot{x} \\\\ \\ddot{y} \\end{bmatrix}', acc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Same expressions as before.

        We can also use the Jacobian matrix to calculate the torque due to a force on the link:$T \quad=\quad \mathbf{J}^T \begin{bmatrix} F_{ex} \\ F_{ey} \end{bmatrix}$"""
    )
    return


@app.cell
def _(Fex, Fey, J_1, Matrix, eq):
    Te = J_1.T * Matrix((Fex, Fey))
    eq('T_e', Te[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Where in this case we considered that the force was applied to the center of mass, just because we already had the Jacobian calculated at that position.

        We could simulate the movement of this one-link system for a typical human movement to understand the magnitude of these physical quantities.  
        The reader is invited to that now. We will perform this simulation for a two-link system next
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Two-link system

        Let's study the dynamics of a planar double inverted pendulum (see Figure 2) as a model of two interconnected segments in the human body with an external force acting on the distal segment. Once again, we will consider that there are muscles around each joint and they generate torques.

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/invpend2.png?raw=1" alt="Double inverted pendulum"/><figcaption><i><center>Figure 2. Planar double inverted pendulum connected by hinge joints with joint actuators and corresponding free body diagrams. See text for notation convention.</center></i></figcaption>

        The following notation convention will be used for this problem:  
         - Subscript$i$runs 1 or 2 meaning first (most proximal) or second joint when referring to angles, joint moments, or joint reaction forces, or meaning first or second segment when referring to everything else.  
         -$L_i$is the length of segment$i$.  
         -$d_i$is the distance from the proximal joint of segment$i$to its center of mass position.  
         -$m_i$is the mass of segment$i$. 
         -$g$is the gravitational acceleration (+).   
         -$\alpha_i$is the angular position of joint$i$in the joint space,$\ddot{\alpha_i}$is the corresponding angular acceleration.
         -$\theta_i$is the angular position of joint$i$in the segmental space w.r.t. horizontal,$\theta_1=\alpha_1$and$\theta_2=\alpha_1+\alpha_2$.  
         -$I_i$is the moment of inertia of segment$i$around its center of mass position.  
         -$F_{ri}$is the reaction force at joint$i$.  
         -$F_{e}$is the external force acting on the distal segment.
         -$T_i$is the moment of force (torque) at joint$i$.  

        Hence we know we will need the linear accelerations for solving the Newton-Euler equations, let's deduce them first.
        """
    )
    return


@app.cell
def _(Symbol, dynamicsymbols, symbols):
    t_1 = Symbol('t')
    (d1, d2, L1, L2) = symbols('d1, d2, L_1 L_2', positive=True)
    (a1, a2) = dynamicsymbols('alpha1 alpha2')
    (a1d, a2d) = (a1.diff(t_1), a2.diff(t_1))
    (a1dd, a2dd) = (a1.diff(t_1, 2), a2.diff(t_1, 2))
    return L1, L2, a1, a1d, a1dd, a2, a2d, a2dd, d1, d2, t_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Kinematics

        Once again, if the angular positions$\alpha_1(t)$and$\alpha_2(t)$are known, the coordinates$(x_1(t), y_1(t))$and$(x_2(t), y_2(t))$and their derivatives can be readily determined (by forward kinematics):

        #### Link 1
        """
    )
    return


@app.cell
def _(a1, cos, d1, eq, sin, t_1):
    (x1, y1) = (d1 * cos(a1), d1 * sin(a1))
    (x1d, y1d) = (x1.diff(t_1), y1.diff(t_1))
    (x1dd, y1dd) = (x1d.diff(t_1), y1d.diff(t_1))
    eq('x_1', x1)
    eq('\\dot{x_1}', x1d)
    eq('\\ddot{x_1}', x1dd)
    eq('y_1', y1)
    eq('\\dot{y_1}', y1d)
    eq('\\ddot{y_1}', y1dd)
    return x1dd, y1dd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Link 2
        """
    )
    return


@app.cell
def _(L1, a1, a2, cos, d2, eq, sin, t_1):
    (x2, y2) = (L1 * cos(a1) + d2 * cos(a1 + a2), L1 * sin(a1) + d2 * sin(a1 + a2))
    (x2d, y2d) = (x2.diff(t_1), y2.diff(t_1))
    (x2dd, y2dd) = (x2d.diff(t_1), y2d.diff(t_1))
    eq('x_2', x2)
    eq('\\dot{x_2}', x2d)
    eq('\\ddot{x_2}', x2dd)
    eq('y_2', y2)
    eq('\\dot{y_2}', y2d)
    eq('\\ddot{y_2}', y2dd)
    return x2, x2dd, y2, y2dd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Inspecting the equations above, we see a new kind of acceleration, proportional to$\dot{\alpha_1}\dot{\alpha_2}$. This acceleration is due to the [Coriolis effect](http://en.wikipedia.org/wiki/Coriolis_effect) and is  present only when there are movement in the two joints.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Jacobian matrix for the two-link system

        The Jacobian matrix for the two-link system w.r.t. the center of mass of the second link is:
        """
    )
    return


@app.cell
def _(Matrix, a1, a2, eq, x2, y2):
    r2 = Matrix([[x2, y2]])
    J2 = r2.jacobian([a1, a2])
    eq(r'\mathbf{J}', J2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Kinetics

        From the free body diagrams, the Newton-Euler equations for links 1 and 2 of the planar double inverted pendulum are:$\begin{array}{l l}
        F_{r2x} + F_{e,x} &=& m_2\ddot{x}_{2} \\
        \\
        F_{r2y} - m_2g + F_{e,y} &=& m_2\ddot{y}_{2} \\
        \\
        T_2 + d_2F_{r2x}\sin(\alpha_1+\alpha_2) - d_2F_{r2y}\cos(\alpha_1+\alpha_2) - (L_2-d_2)F_{e,x}\sin(\alpha_1+\alpha_2) - (L_2-d_2)F_{e,y}\cos(\alpha_1+\alpha_2) &=& I_{2}(\ddot{\alpha}_1+\ddot{\alpha}_2) \\
        \\
        F_{r1x} - F_{r2x} &=& m_1\ddot{x}_{1} \\
        \\
        F_{r1y} - F_{r2y} - m_1g &=& m_1\ddot{y}_{1} \\
        \\
        T_1 - T_2 + d_1F_{r1x}\sin\alpha_1 - d_1F_{r1y}\cos\alpha_1 + (L_1-d_1)F_{r2x}\sin\alpha_1 - (L_1-d_1)F_{r2y}\cos\alpha_1 &=& I_{1}\ddot{\alpha}_1
        \end{array}$If we want to determine the joint torques and we know the kinematics of the links, the inverse dynamics approach, we isolate the joint torques in the equations above, start solving for link 2 and then link 1. To determine the kinematics knowing the joint torques, the direct dynamics approach, we isolate the joint angular accelerations in the equations above and solve the ordinary differential equations.

        Let's express the equations for the torques substituting the terms we know:
        """
    )
    return


@app.cell
def _(symbols):
    (m1, m2, I1, I2, g_1) = symbols('m_1, m_2, I_1 I_2 g', positive=True)
    (Fex_1, Fey_1) = symbols('F_ex F_ey')
    return Fex_1, Fey_1, I1, I2, g_1, m1, m2


@app.cell
def _(
    Fex_1,
    Fey_1,
    I1,
    I2,
    L1,
    L2,
    a1,
    a1dd,
    a2,
    a2dd,
    cos,
    d1,
    d2,
    g_1,
    m1,
    m2,
    simplify,
    sin,
    x1dd,
    x2dd,
    y1dd,
    y2dd,
):
    Fr2x = m2 * x2dd - Fex_1
    Fr2y = m2 * y2dd + m2 * g_1 - Fey_1
    T2 = I2 * (a1dd + a2dd) - d2 * Fr2x * sin(a1 + a2) + d2 * Fr2y * cos(a1 + a2) + (L2 - d2) * Fex_1 * sin(a1 + a2) - (L2 - d2) * Fey_1 * cos(a1 + a2)
    T2 = simplify(T2)
    Fr1x = m1 * x1dd + Fr2x
    Fr1y = m1 * y1dd + Fr2y + m1 * g_1
    T1 = I1 * a1dd + T2 - d1 * Fr1x * sin(a1) + d1 * Fr1y * cos(a1) - (L1 - d1) * Fr2x * sin(a1) + (L1 - d1) * Fr2y * cos(a1)
    T1 = simplify(T1)
    return T1, T2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The expressions for the joint moments of force (torques) are:
        """
    )
    return


@app.cell
def _(T1, T2, eq):
    eq(r'T_1', T1)
    eq(r'T_2', T2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There is an elegant form to display the equations for the torques using generalized coordinates,$q=[\alpha_1, \alpha_2]^T$and grouping the terms proportional to common quantities in matrices, see for example, Craig (2005, page 180), Pandy (2001), and Zatsiorsky (2002, page 383):$\tau \quad=\quad M(q)\ddot{q} + C(q,\dot{q}) + G(q) + E(q,\dot{q})$Where, for this two-link system:  
        -$\tau$is a matrix (2x1) of joint torques;  
        -$M$is the mass or inertia matrix (2x2);  
        -$\ddot{q}$is a matrix (2x1) of angular accelerations;  
        -$C$is a matrix (2x1) of [centipetal](http://en.wikipedia.org/wiki/Centripetal_force) and [Coriolis](http://en.wikipedia.org/wiki/Coriolis_effect) torques;  
        -$G$is a matrix (2x1) of  gravitational torques;  
        -$E$is a matrix (2x1) of external torques.   

        Let's use Sympy to display the equations for the torques in this new form:
        """
    )
    return


@app.cell
def _(T1, T2, a1, a1d, a1dd, a2, a2d, a2dd, dynamicsymbols, t_1):
    (T1_1, T2_1) = (T1.expand(), T2.expand())
    (q1, q2) = dynamicsymbols('q_1 q_2')
    (q1d, q2d) = (q1.diff(t_1), q2.diff(t_1))
    (q1dd, q2dd) = (q1.diff(t_1, 2), q2.diff(t_1, 2))
    T1_1 = T1_1.subs({a1: q1, a2: q2, a1d: q1d, a2d: q2d, a1dd: q1dd, a2dd: q2dd})
    T2_1 = T2_1.subs({a1: q1, a2: q2, a1d: q1d, a2d: q2d, a1dd: q1dd, a2dd: q2dd})
    return T1_1, T2_1, q1, q1d, q1dd, q2, q2d, q2dd


@app.cell
def _(
    Fex_1,
    Fey_1,
    Math,
    Matrix,
    T1_1,
    T2_1,
    display,
    g_1,
    mlatex,
    q1d,
    q1dd,
    q2d,
    q2dd,
    simplify,
):
    M = Matrix(((simplify(T1_1.coeff(q1dd)), simplify(T1_1.coeff(q2dd))), (simplify(T2_1.coeff(q1dd)), simplify(T2_1.coeff(q2dd)))))
    C = Matrix((simplify(T1_1.coeff(q1d ** 2) * q1d ** 2 + T1_1.coeff(q2d ** 2) * q2d ** 2 + T1_1.coeff(q1d * q2d) * q1d * q2d), simplify(T2_1.coeff(q1d ** 2) * q1d ** 2 + T2_1.coeff(q2d ** 2) * q2d ** 2 + T2_1.coeff(q1d * q2d) * q1d * q2d)))
    G_1 = Matrix((simplify(T1_1.coeff(g_1) * g_1), simplify(T2_1.coeff(g_1) * g_1)))
    E = Matrix((simplify(T1_1.coeff(Fex_1) * Fex_1 + T1_1.coeff(Fey_1) * Fey_1), simplify(T2_1.coeff(Fex_1) * Fex_1 + T2_1.coeff(Fey_1) * Fey_1)))
    display(Math('\\begin{eqnarray}\\tau&\\quad=\\quad&\\begin{bmatrix}\\tau_1\\\\ \\tau_2\\\\ \\end{bmatrix} \\\\' + 'M(q)&\\quad=\\quad&' + mlatex(M) + '\\\\' + '\\ddot{q}&\\quad=\\quad&' + mlatex(Matrix((q1dd, q2dd))) + '\\\\' + 'C(q,\\dot{q})&\\quad=\\quad&' + mlatex(C) + '\\\\' + 'G(q)&\\quad=\\quad&' + mlatex(G_1) + '\\\\' + 'E(q,\\dot{q})&\\quad=\\quad&' + mlatex(E) + '\\end{eqnarray}'))
    return C, E, G_1, M


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With this convention, to perform inverse dynamics we would calculate:$\tau \quad=\quad M(q)\ddot{q} + C(q,\dot{q}) + G(q) + E(q,\dot{q})$And for direct dynamics we would solve the differential equation:$\ddot{q} \quad=\quad M(q)^{-1} \left[\tau - C(q,\dot{q}) - G(q) - E(q,\dot{q}) \right]$The advantage of calculating analytically the derivatives of the position vector as function of the joint angles and using the notation above is that each term that contributes to each joint torque or acceleration can be easily identified. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Coupling (or interaction) effects

        The two terms off the main diagonal in the inertia matrix (which are the same) and the centripetal and Coriolis terms represent the effects of the movement (nonzero velocity) of one joint over the other. These torques are referred as coupling or interaction effects (see for example Hollerbach and Flash (1982) for an application of this concept in the study of the motor control of the upper limb movement).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Planar double pendulum

        Using the same equations above, one can represent a planar double pendulum (hanging from the top, not inverted) considering the angles$\alpha_1$and$\alpha_2$negative, e.g., at$\alpha_1=-90^o$and$\alpha_2=0$the pendulum is hanging vertical.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### WARNING:$F_r$is not the actual joint reaction force!

        For these two examples, in the Newton-Euler equations based on the free body diagrams we represented the consequences of all possible muscle forces on a joint as a net muscle torque and all forces acting on a joint as a resultant joint reaction force. That is, all forces between segments were represented as a resultant force that doesn't generate torque and a force couple (or free moment) that only generates torque. This is an important principle in mechanics of rigid bodies, see for example [this text](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/FreeBodyDiagram.ipynb). However, this principle creates the unrealistic notion that the sum of forces is applied directly on the joint (which has no further implication for a rigid body), but it is inaccurate for the understanding of the local effects on the joint. So, if we are trying to understand the stress on the joint or mechanisms of joint injury, the forces acting on the joint and on the rest of the segment must be considered individually.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Determination of muscle force

        The torque$T$exerted by a muscle is given by the product between the muscle-tendon moment arm$r$and its force$F$. For the human body, there is more than one muscle crossing a joint and several joints. In such case, the torques due to the muscles are expressed in the following matrix form considering$n$joints and$m$muscles:

        \begin{eqnarray}
        \begin{bmatrix} T_1 \\ \vdots \\ T_n \end{bmatrix} &\quad=\quad& \begin{bmatrix} r_{11} & \cdots & r_{1m} \\ \vdots & \ddots & \vdots \\ r_{n1} & \cdots & r_{nm} \end{bmatrix} \begin{bmatrix} F_1 \\ \vdots \\ F_m \end{bmatrix}
        \label{}
        \end{eqnarray}

        Where$r_{nm}$is the moment arm about joint$n$of the muscle$m$.  
        In the example of the two-link system, we sketched two uniarticular muscles for each of the two joints, consequently:  

        \begin{eqnarray}
        \begin{bmatrix} T_1 \\ T_2 \end{bmatrix} &\quad=\quad& \begin{bmatrix} r_{1,ext} & -r_{1,flex} & 0 & 0 \\ 0 & 0 & r_{1,ext} & -r_{1,flex} \end{bmatrix} \begin{bmatrix} F_{1,ext} \\ F_{1,flex} \\ F_{2,ext} \\ F_{2,flex} \end{bmatrix}
        \label{}
        \end{eqnarray} 

        Note the opposite signs for the moment arms of the extension and flexion muscles hence they generate opposite torques. We could have represented the opposite signs in the muscle forces instead of in the moment arms.

        The moment arm of a muscle varies with the motion of the joints it crosses. In this case, using the [virtual work principle](http://en.wikipedia.org/wiki/Virtual_work) the moment arm can be given by (Sherman et al., 2013; Nigg and Herzog, 2006, page 634):$r(q) \quad=\quad \dfrac{\partial L_{MT}(q)}{\partial q}$Where$L_{MT}(q)$is the length of the muscle-tendon unit expressed as a function of angle$q$.

        For the simulation of human movement, muscles can be modeled as [Hill-type muscles](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/MuscleSimulation.ipynb), the torques they generate are given by the matrix above, and this matrix is entered in the ODE for a multibody system dynamics we deduced before:$\ddot{q} \quad=\quad M(q)^{-1} \left[R_{MT}(q)F_{MT}(a,L_{MT},\dot{L}_{MT}) - C(q,\dot{q}) - G(q) - E(q,\dot{q}) \right]$Where$R_{MT}$and$F_{MT}$are matrices for the moment arms and muscle-tendon forces, respectively.
        This ODE is then solved numerically given initial values; but this problem is far from trivial for a simulation with several segments and muscles.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Numerical simulation of inverse dynamics

        Let's simulate a voluntary movement of the upper limb using the planar two-link system as a model in order to visualize the contribution of each torque term.  

        We will ignore the muscle dynamics and we will calculate the joint torques necessary to move the upper limb from one point to another under the assumption that the movement is performed with the smoothest trajectory possible. I.e., the movement is performed with a [minimum-jerk trajectory](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/MinimumJerkHypothesis.ipynb), a hypothesis about control of voluntary movements proposed by Flash and Hogan (1985).

        Once we determine the desired trajectory, we can calculate the velocity and acceleration of the segments and combine with anthropometric measures to calculate the joint torques necessary to move the segments. This means we will perform inverse dynamics. 

        Let's simulate a slow (4 s) and a fast (0.5 s) movement of the upper limb starting at the anatomical neutral position (upper limb at the side of the trunk) and ending with the upper arm forward at horizontal and elbow flexed at 90 degrees.

        First, let's import the necessary Python libraries and customize the environment:
        """
    )
    return


@app.cell
def _():
    import numpy as np
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['lines.linewidth'] = 3
    matplotlib.rcParams['font.size'] = 13
    matplotlib.rcParams['lines.markersize'] = 5
    matplotlib.rc('axes', grid=True, labelsize=14, titlesize=16, ymargin=0.05)
    matplotlib.rc('legend', numpoints=1, fontsize=11)
    import sys
    sys.path.insert(1, r'./../functions')  # add to pythonpath
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's take the anthropometric data from Dempster's model (see [Body segment parameters](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/BodySegmentParameters.ipynb)):
        """
    )
    return


@app.cell
def _():
    height, mass = 1.70,               70  # m, kg
    L1n, L2n     = 0.188*height,       0.253*height
    d1n, d2n     = 0.436*L1n,          0.682*L2n
    m1n, m2n     = 0.0280*mass,        0.0220*mass
    rg1n, rg2n   = 0.322,              0.468
    I1n, I2n     = m1n*(rg1n*L1n)**2,  m2n*(rg2n*L2n)**2
    return I1n, I2n, L1n, L2n, d1n, d2n, m1n, m2n


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Considering these lengths, the initial and final positions of the endpoint (finger tip) for the simulated movement will be:
        """
    )
    return


@app.cell
def _(L1n, L2n):
    xi, yi = 0, -L1n-L2n
    xf, yf = L1n, L2n
    gn = 9.81  # gravity acceleration m/s2
    return gn, xf, xi, yf, yi


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Slow movement
        """
    )
    return


@app.cell
def _():
    duration = 4  # seconds
    return (duration,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The endpoint minimum jerk trajectory will be (see [Kinematic chain in a plane (2D)](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/KinematicChain.ipynb)):
        """
    )
    return


@app.cell
def _():
    from minjerk import minjerk
    return (minjerk,)


@app.cell
def _(duration, minjerk, xf, xi, yf, yi):
    (time, rlin, _vlin, _alin, _jlin) = minjerk([xi, yi], [xf, yf], duration=duration)
    return rlin, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's find the joint angles to produce this minimum-jerk trajectory (inverse kinematics):
        """
    )
    return


@app.cell
def _():
    from invkin2_2d import invkin
    return (invkin,)


@app.cell
def _(L1n, L2n, invkin, rlin, time):
    rang = invkin(time, rlin, L1=L1n, L2=L2n)
    return (rang,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For the joint torques, we need to calculate the angular velocity and acceleration. Let's do that using numerical differentiation:
        """
    )
    return


@app.cell
def _(duration, np, plt, rang, time):
    def diff_c(ang, duration):
        """Numerical differentiations using the central difference for the angular data.
        """
        # central difference (f(x+h)-f(x-h))/(2*h)
        dt = duration/(ang.shape[0]-1)
        vang = np.empty_like(rang)
        aang = np.empty_like(rang)
        vang[:, 0] = np.gradient(rang[:, 0], dt)
        vang[:, 1] = np.gradient(rang[:, 1], dt)
        aang[:, 0] = np.gradient(vang[:, 0], dt)
        aang[:, 1] = np.gradient(vang[:, 1], dt)
    
        _, ax = plt.subplots(1, 3, sharex=True, figsize=(10, 3))
        ax[0].plot(time, rang*180/np.pi)
        ax[0].legend(['Ang 1', 'Ang 2'], framealpha=.5, loc='best')
        ax[1].plot(time, vang*180/np.pi)
        ax[2].plot(time, aang*180/np.pi)
        ylabel = [r'Displacement [$\mathrm{^o}$]', r'Velocity [$\mathrm{^o/s}$]',
                  r'Acceleration [$\mathrm{^o/s^2}$]']
        for i, axi in enumerate(ax):
            axi.set_xlabel('Time [$s$]')
            axi.set_ylabel(ylabel[i])
            axi.xaxis.set_major_locator(plt.MaxNLocator(4))
            axi.yaxis.set_major_locator(plt.MaxNLocator(4))
        plt.tight_layout()
        plt.show()

        return vang, aang

    vang, aang = diff_c(rang, duration)
    return aang, diff_c, vang


@app.cell
def _(Fex_1, Fey_1, I1, I2, L1, L2, d1, d2, g_1, m1, m2, plt, t_1):
    def dyna(time, L1n, L2n, d1n, d2n, m1n, m2n, gn, I1n, I2n, q1, q2, rang, vang, aang, Fexn, Feyn, M, C, G, E):
        """Numerical calculation and plot for the torques of a planar two-link system.
        """
        from sympy import lambdify, symbols
        Mfun = lambdify((I1, I2, L1, L2, d1, d2, m1, m2, q1, q2), M, 'numpy')
        Mn = Mfun(I1n, I2n, L1n, L2n, d1n, d2n, m1n, m2n, rang[:, 0], rang[:, 1])
        _M00 = Mn[0, 0] * aang[:, 0]
        _M01 = Mn[0, 1] * aang[:, 1]
        _M10 = Mn[1, 0] * aang[:, 0]
        _M11 = Mn[1, 1] * aang[:, 1]
        (Q1d, Q2d) = symbols('Q1d Q2d')
        dicti = {q1.diff(t_1, 1): Q1d, q2.diff(t_1, 1): Q2d}
        C0fun = lambdify((L1, d2, m2, q2, Q1d, Q2d), C[0].subs(dicti), 'numpy')
        _C0 = C0fun(L1n, d2n, m2n, rang[:, 1], vang[:, 0], vang[:, 1])
        C1fun = lambdify((L1, d2, m2, q2, Q1d, Q2d), C[1].subs(dicti), 'numpy')
        _C1 = C1fun(L1n, d2n, m2n, rang[:, 1], vang[:, 0], vang[:, 1])
        G0fun = lambdify((L1, d1, d2, m1, m2, g_1, q1, q2), G[0], 'numpy')
        _G0 = G0fun(L1n, d1n, d2n, m1n, m2n, gn, rang[:, 0], rang[:, 1])
        G1fun = lambdify((L1, d1, d2, m1, m2, g_1, q1, q2), G[1], 'numpy')
        _G1 = G1fun(L1n, d1n, d2n, m1n, m2n, gn, rang[:, 0], rang[:, 1])
        E0fun = lambdify((L1, L2, q1, q2, Fex_1, Fey_1), E[0], 'numpy')
        _E0 = E0fun(L1n, L2n, rang[:, 0], rang[:, 1], 0, 0)
        E1fun = lambdify((L1, L2, q1, q2, Fex_1, Fey_1), E[1], 'numpy')
        _E1 = E1fun(L1n, L2n, rang[:, 0], rang[:, 1], Fexn, Feyn)
        (fig, ax) = plt.subplots(1, 2, sharex=True, squeeze=True, figsize=(10, 4))
        ax[0].plot(time, _M00 + _M01)
        ax[0].plot(time, _C0)
        ax[0].plot(time, _G0)
        ax[0].plot(time, _E0)
        ax[0].set_ylabel('Torque [Nm]')
        ax[0].set_title('Joint 1')
        ax[1].plot(time, _M10 + _M11, label='Mass/Inertia')
        ax[1].plot(time, _C1, label='Centripetal/Coriolis       ')
        ax[1].plot(time, _G1, label='Gravitational')
        ax[1].plot(time, _E1, label='External')
        ax[1].set_title('Joint 2')
        fig.legend(framealpha=0.5, bbox_to_anchor=(1.15, 0.95), fontsize=12)
        for (i, axi) in enumerate(ax):
            axi.set_xlabel('Time [$s$]')
            axi.xaxis.set_major_locator(plt.MaxNLocator(4))
            axi.yaxis.set_major_locator(plt.MaxNLocator(4))
        plt.show()
        return (_M00, _M01, _M10, _M11, _C0, _C1, _G0, _G1, _E0, _E1)
    return (dyna,)


@app.cell
def _(
    C,
    E,
    G_1,
    I1n,
    I2n,
    L1n,
    L2n,
    M,
    aang,
    d1n,
    d2n,
    dyna,
    gn,
    m1n,
    m2n,
    q1,
    q2,
    rang,
    time,
    vang,
):
    (Fexn, Feyn) = (0, 0)
    (_M00, _M01, _M10, _M11, _C0, _C1, _G0, _G1, _E0, _E1) = dyna(time, L1n, L2n, d1n, d2n, m1n, m2n, gn, I1n, I2n, q1, q2, rang, vang, aang, Fexn, Feyn, M, C, G_1, E)
    T1a = _M00 + _M01 + _C0 + _G0 + _E0
    T2a = _M10 + _M11 + _C1 + _G1 + _E1
    return Fexn, Feyn


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The joint torques essentially compensate the gravitational torque.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Fast movement 

        Let's see what is changed for a fast movement:
        """
    )
    return


@app.cell
def _(
    C,
    E,
    Fexn,
    Feyn,
    G_1,
    I1n,
    I2n,
    L1n,
    L2n,
    M,
    d1n,
    d2n,
    diff_c,
    dyna,
    gn,
    invkin,
    m1n,
    m2n,
    minjerk,
    q1,
    q2,
    xf,
    xi,
    yf,
    yi,
):
    duration_1 = 0.5
    (time_1, rlin_1, _vlin, _alin, _jlin) = minjerk([xi, yi], [xf, yf], duration=duration_1)
    rang_1 = invkin(time_1, rlin_1, L1=L1n, L2=L2n)
    (vang_1, aang_1) = diff_c(rang_1, duration_1)
    (_M00, _M01, _M10, _M11, _C0, _C1, _G0, _G1, _E0, _E1) = dyna(time_1, L1n, L2n, d1n, d2n, m1n, m2n, gn, I1n, I2n, q1, q2, rang_1, vang_1, aang_1, Fexn, Feyn, M, C, G_1, E)
    _T1b = _M00 + _M01 + _C0 + _G0 + _E0
    _T2b = _M10 + _M11 + _C1 + _G1 + _E1
    return aang_1, rang_1, time_1, vang_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The interaction torques are larger than the gravitational torques for most part of the movement.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Fast movement in the horizontal plane

        Let's simulate a fast movement in the horizontal plane:
        """
    )
    return


@app.cell
def _(
    C,
    E,
    Fexn,
    Feyn,
    G_1,
    I1n,
    I2n,
    L1n,
    L2n,
    M,
    aang_1,
    d1n,
    d2n,
    dyna,
    m1n,
    m2n,
    q1,
    q2,
    rang_1,
    time_1,
    vang_1,
):
    gn_1 = 0
    (_M00, _M01, _M10, _M11, _C0, _C1, _G0, _G1, _E0, _E1) = dyna(time_1, L1n, L2n, d1n, d2n, m1n, m2n, gn_1, I1n, I2n, q1, q2, rang_1, vang_1, aang_1, Fexn, Feyn, M, C, G_1, E)
    _T1b = _M00 + _M01 + _C0 + _G0 + _E0
    _T2b = _M10 + _M11 + _C1 + _G1 + _E1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Numerical simulation of direct dynamics

        Remember that for direct dynamics we want to solve the following differential equation:$\ddot{q} \quad=\quad M(q)^{-1} \left[\tau - C(q,\dot{q}) - G(q) - E(q,\dot{q}) \right]$Let's use the Euler method for solving this equation numerically.  

        First, transforming the equation above into a system of two first-order ODE:$\left\{
        \begin{array}{l l}
        \dfrac{\mathrm{d} q}{\mathrm{d}t} &=& \dot{q}, \quad &q(t_0) = q_0
        \\
        \dfrac{\mathrm{d} \dot{q}}{\mathrm{d}t} &=& M(q)^{-1} \left[\tau - C(q,\dot{q}) - G(q) - E(q,\dot{q}) \right], \quad &\dot{q}(t_0) = \dot{q}_0
        \end{array}
        \right.$Second, we would write a function for the calculation of the system states and another function for the Euler method. 

        Third, now joint torques are the input to the system; just to cast out nines, let's choose as input torques the output torques of the inverse dynamics solution we calculated before.  

        Fourth, plot everything. 

        Easy peasy.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exercises

        1. Derive the equations of motion for a single pendulum (not inverted).  
        2. Derive the equations of motion for a double pendulum (not inverted).  
        3. For the one-link system, simulate a typical trajectory to calculate the joint torque (i.e., perform inverse dynamics).
        4. For the one-link system, simulate a typical joint torque to calculate the trajectory (i.e., perform direct dynamics).
        5. Consider the double pendulum moving in the horizontal plane and with no external force. Find out the type of movement and which torque terms are changed when:   
          a)$\dot{\alpha}_1=0^o$b)$\alpha_2=0^o$c)$\dot{\alpha}_2=0^o$d)$2\alpha_1+\alpha_2=180^o$(hint: a two-link system with this configuration is called polar manipulator)
        6. Derive the equations of motion and the torque terms using angles in the segmental space$(\theta_1,\,\theta_2)$.  
        7. Run the numerical simulations for the torques with different parameters.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Craig JJ (2005) [Introduction to Robotics: Mechanics and Control](http://books.google.com.br/books?id=MqMeAQAAIAAJ). 3rd Edition. Prentice Hall.  
        - Flash T, Hogan N (1985) [The coordination of arm movements: an experimentally confirmed mathematical model](http://www.jneurosci.org/cgi/reprint/5/7/1688.pdf). Journal of Neuroscience, 5, 1688-1703.   
        - Hollerbach JM, Flash T (1982) [Dynamic interactions between limb segments during planar arm movement](http://link.springer.com/article/10.1007%2FBF00353957). Biological Cybernetics, 44, 67-77.  
        - Nigg BM and Herzog W (2006) [Biomechanics of the Musculo-skeletal System](https://books.google.com.br/books?id=hOIeAQAAIAAJ&dq=editions:ISBN0470017678). 3rd Edition. Wiley.  
        - Pandy MG (2001) [Computer modeling and simulation](https://drive.google.com/open?id=0BxbW72zV7WmUbXZBR2VRMnF5UTA&authuser=0). Annu. Rev. Biomed. Eng., 3, 245–73.  
        - Sherman MA, Seth A, Delp SL (2013) [What is a moment arm? Calculating muscle effectiveness in biomechanical models using generalized coordinates](http://simtk-confluence.stanford.edu:8080/download/attachments/3376330/ShermanSethDelp-2013-WhatIsMuscleMomentArm-Final2-DETC2013-13633.pdf?version=1&modificationDate=1369103515834) in Proc. ASME Int. Design Engineering Technical Conferences (IDETC), Portland, OR, USA.  
        - Zajac FE (1993) [Muscle coordination of movement: a perspective](http://e.guigon.free.fr/rsc/article/Zajac93.pdf). J Biomech., 26, Suppl 1:109-24.  
        - Zajac FE, Gordon ME (1989) [Determining muscle's force and action in multi-articular movement](https://drive.google.com/open?id=0BxbW72zV7WmUcC1zSGpEOUxhWXM&authuser=0). Exercise and Sport Sciences Reviews, 17, 187-230.  
        - Zatsiorsky VM (2002) [Kinetics of human motion](http://books.google.com.br/books?id=wp3zt7oF8a0C). Human Kinetics.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
