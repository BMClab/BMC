import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Velocity and Acceleration of a point of a rigid body

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
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Python-setup" data-toc-modified-id="Python-setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Python setup</a></span></li><li><span><a href="#Frame-of-reference-attached-to-a-body" data-toc-modified-id="Frame-of-reference-attached-to-a-body-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Frame of reference attached to a body</a></span></li><li><span><a href="#Position-of-a-point-on-a-rigid-body" data-toc-modified-id="Position-of-a-point-on-a-rigid-body-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Position of a point on a rigid body</a></span></li><li><span><a href="#Translation-of-a-rigid-body" data-toc-modified-id="Translation-of-a-rigid-body-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Translation of a rigid body</a></span></li><li><span><a href="#Angular-velocity-of-a-body" data-toc-modified-id="Angular-velocity-of-a-body-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Angular velocity of a body</a></span></li><li><span><a href="#Velocity-of-a-point-with-no-translation" data-toc-modified-id="Velocity-of-a-point-with-no-translation-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Velocity of a point with no translation</a></span></li><li><span><a href="#Relative-velocity-of-a-point-on-a-rigid-body-to-another-point" data-toc-modified-id="Relative-velocity-of-a-point-on-a-rigid-body-to-another-point-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Relative velocity of a point on a rigid body to another point</a></span></li><li><span><a href="#Velocity-of-a-point-on-rigid-body-translating" data-toc-modified-id="Velocity-of-a-point-on-rigid-body-translating-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Velocity of a point on rigid body translating</a></span></li><li><span><a href="#Acceleration-of-a-point-on-a-rigid-body" data-toc-modified-id="Acceleration-of-a-point-on-a-rigid-body-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Acceleration of a point on a rigid body</a></span></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-Internet" data-toc-modified-id="Video-lectures-on-the-Internet-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Video lectures on the Internet</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This notebook shows the expressions of  the velocity and acceleration of a point on rigid body, given the angular velocity of the body.
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


app._unparsable_cell(
    r"""
    import numpy as np
    import matplotlib.pyplot as plt
    !pip install ipympl
    # '%matplotlib widget' command supported automatically in marimo
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import FancyArrowPatch
    """,
    name="_"
)


@app.cell
def _():
    from google.colab import output
    output.enable_custom_widget_manager()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Support for third party widgets will remain active for the duration of the session. To disable support:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Frame of reference attached to a body

        The concept of reference frame in Biomechanics and motor control is very important and central to the understanding of human motion. For example, do we see, plan and control the movement of our hand with respect to reference frames within our body or in the environment we move? Or a combination of both?  
        The figure below, although derived for a robotic system, illustrates well the concept that we might have to deal with multiple coordinate systems.  

        <div class='center-align'><figure><img src="https://raw.githubusercontent.com/demotu/BMC/master/images/coordinatesystems.png" width=450/><figcaption><center><i>Figure. Multiple coordinate systems for use in robots (figure from Corke (2017)).</i></center></figcaption></figure></div>

        For three-dimensional motion analysis in Biomechanics, we may use several different references frames for convenience and refer to them as global, laboratory, local, anatomical, or technical reference frames or coordinate systems (we will study this later).  
        There has been proposed different standardizations on how to define frame of references for the main segments and joints of the human body. For instance, the International Society of Biomechanics has a [page listing standardization proposals](https://isbweb.org/activities/standards) by its standardization committee and subcommittees:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Position of a point on a rigid body

        The description of the position of a point P of a rotating rigid body is given by:

        <span class="notranslate">${\bf\vec{r}_{P/O}} = x_{P/O}^*{\bf\hat{i}'} + y_{P/O}^*{\bf\hat{j}'}$</span>

        where$x_{P/O}^*$and$y_{P/O}^*$are the coordinates of the point P position at a reference state with the versors described as:

        <span class="notranslate">${\bf\hat{i}'} = \cos(\theta){\bf\hat{i}}+\sin(\theta){\bf\hat{j}}$</span>

        <span class="notranslate">${\bf\hat{j}'} = -\sin(\theta){\bf\hat{i}}+\cos(\theta){\bf\hat{j}}$</span>


        <img src="https://github.com/BMClab/BMC/blob/master/images/rotBody.png?raw=1" style="width: 1000000px;">

        Note that the vector${\bf\vec{r}_{P/O}}$has always the same description for any point P of the rigid body when described as a linear combination of <span class="notranslate">${\bf\hat{i}'}$</span> and <span class="notranslate">${\bf\hat{j}'}$</span>.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Translation of a rigid body

         Let's consider now the case in which, besides a rotation, a translation of the body happens. This situation is represented in the figure below. In this case, the position of the point P is given by:

        <span class="notranslate">${\bf\vec{r}_{P/O}} = {\bf\vec{r}_{A/O}}+{\bf\vec{r}_{P/A}}= {\bf\vec{r}_{A/O}}+x_{P/A}^*{\bf\hat{i}'} + y_{P/A}^*{\bf\hat{j}'}$</span>

        <img src="https://github.com/BMClab/BMC/blob/master/images/rotTrBody.png?raw=1" width=1000/>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Angular velocity of a body

        The magnitude of the angular velocity of a rigid  body rotating on a plane is defined as:

        <span class="notranslate">$\omega = \frac{d\theta}{dt}$</span>

        Usually, it is defined an angular velocity vector perpendicular to the plane where the rotation occurs (in this case the x-y plane) and with magnitude$\omega$:

        <span class="notranslate">$\vec{\bf{\omega}} = \omega\hat{\bf{k}}$</span>


        <img src="https://github.com/BMClab/BMC/blob/master/images/angvel.png?raw=1" width=600/>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Velocity of a point with no translation
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        First we will consider the situation with no translation. The velocity of the point P is given by:

        <span class="notranslate">${\bf\vec{v}_{P/O}} = \frac{d{\bf\vec{r}_{P/O}}}{dt} = \frac{d(x_{P/O}^*{\bf\hat{i}'} + y_{P/O}^*{\bf\hat{j}'})}{dt}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To continue this deduction, we have to find the expression of the derivatives of <span class="notranslate">${\bf\hat{i}'}$</span> and <span class="notranslate">${\bf\hat{j}'}$</span>. This is very similar to the derivative expressions of${\bf\hat{e_R}}$and${\bf\hat{e_\theta}}$of [polar basis](http://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/PolarBasis.ipynb).

        <span class="notranslate">$\frac{d{\bf\hat{i}'}}{dt} = -\dot{\theta}\sin(\theta){\bf\hat{i}}+\dot{\theta}\cos(\theta){\bf\hat{j}} = \dot{\theta}{\bf\hat{j}'}$</span>

        <span class="notranslate">$\frac{d{\bf\hat{j}'}}{dt} = -\dot{\theta}\cos(\theta){\bf\hat{i}}-\dot{\theta}\sin(\theta){\bf\hat{j}} = -\dot{\theta}{\bf\hat{i}'}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Another way to represent the expressions above is by using the vector form to express the angular velocity$\dot{\theta}$. It is usual to represent the angular velocity as a vector in the direction${\bf\hat{k}}$:${\bf\vec{\omega}} = \dot{\theta}{\bf\hat{k}} = \omega{\bf\hat{k}}$. Using this definition of the angular velocity, we can write the above expressions as:

        <span class="notranslate">$\frac{d{\bf\hat{i}'}}{dt} = \dot{\theta}{\bf\hat{j}'} = \dot{\theta} {\bf\hat{k}}\times {\bf\hat{i}'} = {\bf\vec{\omega}} \times {\bf\hat{i}'}$</span>

        <span class="notranslate">$\frac{d{\bf\hat{j}'}}{dt} = -\dot{\theta}{\bf\hat{i}'} = \dot{\theta} {\bf\hat{k}}\times {\bf\hat{j}'} ={\bf\vec{\omega}} \times {\bf\hat{j}'}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        So, the velocity of the point P in the situation of no translation is:

        <span class="notranslate">${\bf\vec{v}_{P/O}} = \frac{d(x_{P/O}^*{\bf\hat{i}'} + y_{P/O}^*{\bf\hat{j}'})}{dt} = x_{P/O}^*\frac{d{\bf\hat{i}'}}{dt} + y_{P/O}^*\frac{d{\bf\hat{j}'}}{dt}=x_{P/O}^*{\bf\vec{\omega}} \times {\bf\hat{i}'} + y_{P/O}^*{\bf\vec{\omega}} \times {\bf\hat{j}'} = {\bf\vec{\omega}} \times \left(x_{P/O}^*{\bf\hat{i}'}\right) + {\bf\vec{\omega}} \times \left(y_{P/O}^*{\bf\hat{j}'}\right) ={\bf\vec{\omega}} \times \left(x_{P/O}^*{\bf\hat{i}'}+y_{P/O}^*{\bf\hat{j}'}\right)$</span>

        <span class="notranslate">${\bf\vec{v}_{P/O}} = {\bf\vec{\omega}} \times {\bf{\vec{r}_{P/O}}}$</span>

        This expression shows that the velocity vector of any point of a rigid body is orthogonal to the vector linking the point O and the point P.

        It is worth to note that despite the above expression was deduced for a planar movement, the expression above is general, including three dimensional movements.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Relative velocity of a point on a rigid body to another point

        To compute the velocity of a point on a rigid body that is translating, we need to find the expression of the velocity of a point (P) in relation to another point on the body (A). So:

        <span class="notranslate">${\bf\vec{v}_{P/A}} = {\bf\vec{v}_{P/O}}-{\bf\vec{v}_{A/O}} = {\bf\vec{\omega}} \times {\bf{\vec{r}_{P/O}}} - {\bf\vec{\omega}} \times {\bf{\vec{r}_{A/O}}} = {\bf\vec{\omega}} \times ({\bf{\vec{r}_{P/O}}}-{\bf{\vec{r}_{A/O}}}) =  {\bf\vec{\omega}} \times {\bf{\vec{r}_{P/A}}}$</span>


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Velocity of a point on rigid body translating

        The velocity of a point on a rigid body that is translating is given by:

        <span class="notranslate">${\bf\vec{v}_{P/O}}  = \frac{d{\bf\vec{r}_{P/O}}}{dt} = \frac{d({\bf\vec{r}_{A/O}}+x_{P/A}^*{\bf\hat{i}'} + y_{P/A}^*{\bf\hat{j}'})}{dt} = \frac{d{\bf\vec{r}_{A/O}}}{dt}+\frac{d(x_{P/A}^*{\bf\hat{i}'} + y_{P/A}^*{\bf\hat{j}'})}{dt} = {\bf\vec{v}_{A/O}} + {\bf\vec{\omega}} \times {\bf{\vec{r}_{P/A}}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Below is an example of a body rotating with the angular velocity of$\omega = \pi/10$rad/s and translating at the velocity of <span class="notranslate">${\bf\vec{v}} = 0.7 {\bf\hat{i}} + 0.5 {\bf\hat{j}}$m/s</span>. The red arrow indicates the velocity of the geometric center of the body and the blue arrow indicates the velocity of the lower point of the body
        """
    )
    return


@app.cell
def _(FancyArrowPatch, FuncAnimation, np, plt):
    t = np.linspace(0,13,40)
    omega = np.pi/10 #[rad/s]
    voa = np.array([[0.7],[0.5]]) # velocity of center of mass
    fig = plt.figure()
    plt.grid()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("on")
    plt.rcParams['figure.figsize']=5,5
    def run(i):
        ax.clear()
        theta = omega * t[i]
        phi = np.linspace(0,2*np.pi,100)
        B = np.squeeze(np.array([[2*np.cos(phi)],[6*np.sin(phi)]]))
        Baum = np.vstack((B,np.ones((1,np.shape(B)[1]))))
        roa = voa * t[i]
        R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        T = np.vstack((np.hstack((R,roa)), np.array([0,0,1])))
        BRot = R@B
        BRotTr = T@Baum



        plt.plot(BRotTr[0,:],BRotTr[1,:], roa[0], roa[1],'.')
        plt.fill(BRotTr[0,:],BRotTr[1,:], 'g')

        vVoa = FancyArrowPatch(roa.squeeze(), roa.squeeze()+5*voa.squeeze(), mutation_scale=20,
                               lw=2, arrowstyle="->", color="r", alpha=1)
        ax.add_artist(vVoa)

        element = 75

        # cross product between omega and r
        Vp = voa + np.cross(np.array([0,0,omega]), BRot[:,[element]].T)[:,0:2].T

        vVP = FancyArrowPatch(BRotTr[0:2,element], BRotTr[0:2,element] + 5*Vp.squeeze(),
                              mutation_scale=20,
                              lw=2, arrowstyle="->", color="b", alpha=1)
        ax.add_artist(vVP)

        plt.xlim((-10, 20))
        plt.ylim((-10, 20))

    ani = FuncAnimation(fig, run, frames = 50, repeat=False,  interval =500)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Acceleration of  a point on a rigid body

        The acceleration of a point on a rigid body is obtained by deriving the previous expression:

        <span class="notranslate">
        \begin{align}
        {\bf\vec{a}_{P/O}} =& {\bf\vec{a}_{A/O}} + \dot{\bf\vec{\omega}} \times {\bf{\vec{r}_{P/A}}} + {\bf\vec{\omega}} \times {\bf{\vec{v}_{P/A}}} =\\
            =&{\bf\vec{a}_{A/O}} + \dot{\bf\vec{\omega}} \times {\bf{\vec{r}_{P/A}}} + {\bf\vec{\omega}} \times ({\bf\vec{\omega}} \times {\bf{\vec{r}_{P/A}}}) =\\
            =&{\bf\vec{a}_{A/O}} + \ddot{\theta}\bf\hat{k} \times {\bf{\vec{r}_{P/A}}} - \dot{\theta}^2{\bf{\vec{r}_{P/A}}}
        \end{align}
        </span>

        The acceleration has three terms:

        - <span class="notranslate">${\bf\vec{a}_{A/O}}$</span> -- the acceleration of the point O.
        - <span class="notranslate">$\ddot{\theta}\bf\hat{k} \times {\bf{\vec{r}_{P/A}}}$</span> -- the acceleration of the point P due to the angular acceleration of the body.
        - <span class="notranslate">$- \dot{\theta}^2{\bf{\vec{r}_{P/A}}}$</span> -- the acceleration of the point P due to the angular velocity of the body. It is known as centripetal acceleration.



        Below is an example of a rigid  body with an angular acceleration of <span class="notranslate">$\alpha = \pi/150$rad/s$^2$</span> and initial angular velocity of <span class="notranslate">$\omega_0 = \pi/100$rad/s</span>. Consider also that the center of the body accelerates with <span class="notranslate">${\bf\vec{a}} = 0.01{\bf\hat{i}} + 0.05{\bf\hat{j}}$</span>, starting from rest.

        """
    )
    return


@app.cell
def _(FancyArrowPatch, FuncAnimation, np, plt):
    t_1 = np.linspace(0, 20, 40)
    alpha = np.pi / 150
    omega0 = np.pi / 100
    aoa = np.array([[0.01], [0.05]])
    fig_1 = plt.figure()
    plt.grid()
    ax_1 = fig_1.add_axes([0, 0, 1, 1])
    ax_1.axis('on')
    plt.rcParams['figure.figsize'] = (5, 5)
    theta = 0
    omega_1 = 0

    def run_1(i):
        ax_1.clear()
        phi = np.linspace(0, 2 * np.pi, 100)
        B = np.squeeze(np.array([[2 * np.cos(phi)], [6 * np.sin(phi)]]))
        Baum = np.vstack((B, np.ones((1, np.shape(B)[1]))))
        omega = alpha * t_1[i] + omega0
        theta = alpha / 2 * t_1[i] ** 2 + omega0 * t_1[i]
        voa = aoa * t_1[i]
        roa = aoa / 2 * t_1[i] ** 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        T = np.vstack((np.hstack((R, roa)), np.array([0, 0, 1])))
        BRot = R @ B
        BRotTr = T @ Baum
        plt.plot(BRotTr[0, :], BRotTr[1, :], roa[0], roa[1], '.')
        plt.fill(BRotTr[0, :], BRotTr[1, :], 'g')
        element = 75
        ap = aoa + np.cross(np.array([0, 0, alpha]), BRot[:, [element]].T)[:, 0:2].T - omega ** 2 * BRot[:, [element]]
        vVP = FancyArrowPatch(BRotTr[0:2, element], BRotTr[0:2, element] + 5 * ap.squeeze(), mutation_scale=20, lw=2, arrowstyle='->', color='b', alpha=1)
        ax_1.add_artist(vVP)
        plt.xlim((-10, 20))
        plt.ylim((-10, 20))
    ani_1 = FuncAnimation(fig_1, run_1, frames=50, repeat=False, interval=500)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

        - Read pages 958-971 of the 18th chapter of the [Ruina and Rudra's book] (http://ruina.tam.cornell.edu/Book/index.html) about circular motion of particle.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the Internet

        - [Kinematics Of Rigid Bodies - General Plane Motion - Solved Problems](https://www.youtube.com/watch?v=4LsLy9iJKFA)
        - [Kinematics of Rigid Bodies -Translation And Rotation About Fixed Axis - Rectilinear and Rotational](https://www.youtube.com/watch?v=VnzsQmP6eMQ)  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        - 1. Solve the problems 16.2.5, 16.2.10, 16.2.11 and 16.2.20 from [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html).
        - 2. Solve the problems 17.1.2, 17.1.8, 17.1.9, 17.1.10, 17.1.11 and 17.1.12 from [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html).
        - 3. Solve the problems 2.1, 2.2, 2.7, 2.8, 2.17
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Reference

        - Ruina A, Rudra P (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.
        - Corke P (2017) [Robotics, Vision and Control: Fundamental Algorithms in MATLAB](http://www.petercorke.com/RVC/). 2nd ed. Springer-Verlag Berlin.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
