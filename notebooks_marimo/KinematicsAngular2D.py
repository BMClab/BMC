import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a href="https://colab.research.google.com/github/BMClab/BMC/blob/master/notebooks/KinematicsAngular2D.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Angular kinematics in a plane (2D)

        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br)    
        > Federal University of ABC, Brazil

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Python-setup" data-toc-modified-id="Python-setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Python setup</a></span></li><li><span><a href="#Angles-in-a-plane" data-toc-modified-id="Angles-in-a-plane-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Angles in a plane</a></span></li><li><span><a href="#Angle-between-two-3D-vectors" data-toc-modified-id="Angle-between-two-3D-vectors-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Angle between two 3D vectors</a></span></li><li><span><a href="#Angular-position-velocity-and-acceleration" data-toc-modified-id="Angular-position-velocity-and-acceleration-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Angular position, velocity, and acceleration</a></span><ul class="toc-item"><li><span><a href="#The-antiderivative" data-toc-modified-id="The-antiderivative-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>The antiderivative</a></span></li></ul></li><li><span><a href="#Relationship-between-linear-and-angular-kinematics" data-toc-modified-id="Relationship-between-linear-and-angular-kinematics-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Relationship between linear and angular kinematics</a></span></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-Internet" data-toc-modified-id="Video-lectures-on-the-Internet-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Video lectures on the Internet</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Human motion is a combination of linear and angular movement and occurs in the three-dimensional (3D) space. For certain movements and depending on the desired or needed degree of detail for the motion analysis, it's possible to perform a two-dimensional (2D, planar) analysis at the main plane of movement. Such simplification is appreciated because the instrumentation and analysis are much more complicated in order to measure 3D motion than for the 2D case.
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
    # Import the necessary libraries
    import numpy as np
    from IPython.display import display, Latex
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2, "lines.markersize": 8})
    return Latex, display, matplotlib, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Angles in a plane
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For the planar case, the calculation of angles is reduced to the application of trigonometry to the kinematic data. For instance, given the coordinates in a plane of markers on a segment as shown in the figure below, the angle of the segment can be calculated using the inverse function of <span class="notranslate">$\sin,\cos$</span>, or <span class="notranslate">$\tan$</span>.

        <div class='center-align'><figure><img src="https://github.com/BMClab/BMC/blob/master/images/segment.png?raw=1" width=250/><figcaption><figcaption><center><i>Figure. A segment in a plane and its coordinates.</i></center></figcaption> </figure></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For better numerical accuracy (and also to distinguish values in the whole quadrant), the inverse function of <span class="notranslate">$\tan$</span> is preferred.   
        For the data shown in the previous figure:   

        <span class="notranslate">$\theta = arctan\left(\frac{y_2-y_1}{x_2-x_1}\right)$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In computer programming (here, Python/Numpy) this is calculated using: `numpy.arctan((y2-y1)/(x2-x1))`. However, for the previous figure the function `arctan` can not distinguish if the segment is at 45o or at 225o, `arctan` will return the same value. Because this, the function `numpy.arctan2(y, x)` is used, but be aware that `arctan2` will return angles between <span class="notranslate">$[-\pi,\pi]$: </span>
        """
    )
    return


@app.cell
def _(Latex, display, np):
    (_x1, _y1) = (0, 0)
    (_x2, _y2) = (1, 1)
    display(Latex('Segment\\;at\\;45^o:'))
    angs = [np.arctan((1 - 0) / (1 - 0)) * 180 / np.pi, np.arctan2(1 - 0, 1 - 0) * 180 / np.pi]
    display(Latex('Using\\;arctan: ' + str(angs[0]) + '\\;Using\\;arctan2: ' + str(angs[1])))
    display(Latex('Segment\\;at\\;225^o:'))
    angs = [np.arctan((-1 - 0) / (-1 - 0)) * 180 / np.pi, np.arctan2(-1 - 0, -1 - 0) * 180 / np.pi]
    display(Latex('Using\\;arctan: ' + str(angs[0]) + '\\;Using\\;arctan2: ' + str(angs[1])))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And Numpy has a function to convert an angle in rad to degrees and the other way around:
        """
    )
    return


@app.cell
def _(np):
    print('np.rad2deg(np.pi/2) =', np.rad2deg(np.pi))
    print('np.deg2rad(180) =', np.deg2rad(180))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's simulate a 2D motion of the arm performing two complete turns around the shoulder to exemplify the use of `arctan2`:
        """
    )
    return


@app.cell
def _(np, plt):
    t = np.arange(0, 2, 0.01)
    x = np.cos(2 * np.pi * t)
    y = np.sin(2 * np.pi * t)
    _ang = np.arctan2(y, x) * 180 / np.pi
    plt.figure(figsize=(12, 4))
    hax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
    hax1.plot(x, y, 'go')
    hax1.plot(0, 0, 'go')
    hax1.set_xlabel('x [m]')
    hax1.set_ylabel('y [m]')
    hax1.set_xlim([-1.1, 1.1])
    hax1.set_ylim([-1.1, 1.1])
    hax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    hax2.plot(t, x, 'bo', label='x')
    hax2.plot(t, y, 'ro', label='y')
    hax2.legend(numpoints=1, frameon=True, framealpha=0.8)
    hax2.set_ylabel('Position [m]')
    hax2.set_ylim([-1.1, 1.1])
    hax3 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    hax3.plot(t, _ang, 'go')
    hax3.set_yticks(np.arange(-180, 181, 90))
    hax3.set_xlabel('Time [s]')
    hax3.set_ylabel('Angle [ o]')
    plt.tight_layout()
    return t, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Because the output of the `arctan2` is bounded to$[-\pi,\pi]$, the angle measured appears chopped in the figure. This problem can be solved using the function `numpy.unwrap`, which detects sudden jumps in the angle and corrects that:
        """
    )
    return


@app.cell
def _(np, plt, t, x, y):
    _ang = np.unwrap(np.arctan2(y, x)) * 180 / np.pi
    (_hfig, _hax) = plt.subplots(1, 1, figsize=(8, 3))
    _hax.plot(t, _ang, 'go')
    _hax.set_yticks(np.arange(start=0, stop=721, step=90))
    _hax.set_xlabel('Time [s]')
    _hax.set_ylabel('Angle [ o]')
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If now we want to measure the angle of a joint (i.e., the angle of a segment in relation to other segment) we just have to subtract the two segment angles (but this is correct only if the angles are at the same plane):
        """
    )
    return


@app.cell
def _(matplotlib, np, plt):
    (_x1, _y1) = (0.0, 0.0)
    (_x2, _y2) = (1.0, 1.0)
    (_x3, _y3) = (1.1, 1.0)
    (_x4, _y4) = (2.1, 0.0)
    (_hfig, _hax) = plt.subplots(1, 1, figsize=(8, 3))
    _hax.plot((_x1, _x2), (_y1, _y2), 'b-', (_x1, _x2), (_y1, _y2), 'ro', linewidth=3, markersize=12)
    _hax.add_patch(matplotlib.patches.FancyArrowPatch(posA=(_x1 + np.sqrt(2) / 3, _y1), posB=(_x2 / 3, _y2 / 3), arrowstyle='->,head_length=10,head_width=5', connectionstyle='arc3,rad=0.3'))
    plt.text(1 / 2, 1 / 5, '$\\theta_1$', fontsize=24)
    _hax.plot((_x3, _x4), (_y3, _y4), 'b-', (_x3, _x4), (_y3, _y4), 'ro', linewidth=3, markersize=12)
    _hax.add_patch(matplotlib.patches.FancyArrowPatch(posA=(_x4 + np.sqrt(2) / 3, _y4), posB=(_x4 - 1 / 3, _y4 + 1 / 3), arrowstyle='->,head_length=10,head_width=5', connectionstyle='arc3,rad=0.3'))
    _hax.xaxis.set_ticks((_x1, _x2, _x3, _x4))
    _hax.yaxis.set_ticks((_y1, _y2, _y3, _y4))
    _hax.xaxis.set_ticklabels(('x1', 'x2', 'x3', 'x4'), fontsize=20)
    plt.text(_x4 + 0.2, _y4 + 0.3, '$\\theta_2$', fontsize=24)
    _hax.add_patch(matplotlib.patches.FancyArrowPatch(posA=(_x2 - 1 / 3, _y2 - 1 / 3), posB=(_x3 + 1 / 3, _y3 - 1 / 3), arrowstyle='->,head_length=10,head_width=5', connectionstyle='arc3,rad=0.3'))
    plt.text(_x1 + 0.8, _y1 + 0.35, '$\\theta_J=\\theta_2-\\theta_1$', fontsize=24)
    _hax.set_xlim(min([_x1, _x2, _x3, _x4]) - 0.1, max([_x1, _x2, _x3, _x4]) + 0.5)
    _hax.set_ylim(min([_y1, _y2, _y3, _y4]) - 0.1, max([_y1, _y2, _y3, _y4]) + 0.1)
    _hax.grid(xdata=(0, 1), ydata=(0, 1))
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The joint angle shown above is simply the difference between the adjacent segment angles:
        """
    )
    return


@app.cell
def _(Latex, display, np):
    (_x1, _y1, _x2, _y2) = (0, 0, 1, 1)
    (_x3, _y3, _x4, _y4) = (1.1, 1, 2.1, 0)
    ang1 = np.arctan2(_y2 - _y1, _x2 - _x1) * 180 / np.pi
    ang2 = np.arctan2(_y3 - _y4, _x3 - _x4) * 180 / np.pi
    display(Latex('\\theta_1=\\;' + str(ang1) + '^o'))
    display(Latex('\\theta_2=\\;' + str(ang2) + '^o'))
    display(Latex('\\theta_J=\\;' + str(ang2 - ang1) + '^o'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The following convention is commonly used to describe the knee and ankle joint angles at the sagittal plane (figure from Winter 2005):

        <div class='center-align'><figure><img src='https://github.com/BMClab/BMC/blob/master/images/jointangles.png?raw=1' width=350 alt='Joint angle convention'/> <figcaption><center><i>Figure. Convention for the sagital joint angles of the lower limb (from Winter, 2009).</i></center></figcaption></figure></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Angle between two 3D vectors

        In certain cases, we have access to the 3D coordinates of markers but we just care for the angle between segments in the plane defined by these segments (but if there is considerable movement in different planes, this simple 2D angle might give unexpected results).   
        Consider that `p1` and `p2` are the 3D coordinates of markers placed on segment 1 and `p2` and `p3` are the 3D coordinates of the markers on segment 2.    

        To determine the 2D angle between the segments, one can use the definition of the dot product:

        <span class="notranslate">$\mathbf{a} \cdot \mathbf{b} = ||\mathbf{a}||\:||\mathbf{b}||\:cos(\theta)\;\;\; \Rightarrow \;\;\; angle = arccos\left(\frac{dot(p2-p1,\;p3-p2)}{norm(p2-p1)*norm(p3-p2)\;\;\;\;\;} \right)$</span>

        Or using the definition of the cross product:

        <span class="notranslate">$\mathbf{a} \times \mathbf{b} = ||\mathbf{a}||\:||\mathbf{b}||\:sin(\theta) \;\; \Rightarrow \;\; angle = arcsin\left(\frac{cross(p2-p1,\;p3-p2)}{norm(p2-p1)*norm(p3-p2)\;\;\;\;\;} \right)$</span>

        But because `arctan2` has a better numerical accuracy, combine the dot and cross products, and in Python notation:
        ```python
        angle = np.arctan2(np.linalg.norm(np.cross(p1-p2, p4-p3)), np.dot(p1-p2, p4-p3))
        ```
        See [this notebook](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/ScalarVector.ipynb) for a review on the mathematical functions cross product and scalar product.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can use the formula above for the angle between two 3D vectors to calculate the joint angle even with the 2D vectors we calculated before:
        """
    )
    return


@app.cell
def _(np):
    p1, p2 = np.array([0, 0]),   np.array([1, 1])    # segment 1
    p3, p4 = np.array([1.1, 1]), np.array([2.1, 0])  # segment 2

    angle = np.arctan2(np.linalg.norm(np.cross(p1-p2, p4-p3)), np.dot(p1-p2, p4-p3))*180/np.pi

    print('Joint angle:', '{0:.1f}'.format(angle))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As expected, the same result.  

        In Numpy, if the third components of vectors are zero, we don't even need to type them; Numpy takes care of adding zero as the third component for the cross product.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Angular position, velocity, and acceleration

        The angular position is a vector, its direction is given by the perpendicular axis to the plane where the angular position is described, and the motion if it occurs it's said to occur around this axis.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Angular velocity is the rate (with respect to time) of change of the angular position:

        <span class="notranslate">$\mathbf{\omega}(t) = \frac{\mathbf{\theta}(t_2)-\mathbf{\theta}(t_1)}{t_2-t_1} = \frac{\Delta \mathbf{\theta}}{\Delta t}$</span>

        <span class="notranslate">$\mathbf{\omega}(t) = \frac{d\mathbf{\theta}(t)}{dt}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Angular acceleration is the rate (with respect to time) of change of the angular velocity, which can also be given by the second-order rate of change of the angular position:

        <span class="notranslate">$\mathbf{\alpha}(t) = \frac{\mathbf{\omega}(t_2)-\mathbf{\omega}(t_1)}{t_2-t_1} = \frac{\Delta \mathbf{\omega}}{\Delta t}$</span>

        Likewise, angular acceleration is the first-order derivative of the angular velocity or the second-order derivative of the angular position vector:   

        <span class="notranslate">$\mathbf{\alpha}(t) = \frac{d\mathbf{\omega}(t)}{dt} = \frac{d^2\mathbf{\theta}(t)}{dt^2}$</span>

        The direction of the angular velocity and acceleration vectors is the same as the angular position (perpendicular to the plane of rotation) and the sense is given by the right-hand rule.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The antiderivative

        As the angular acceleration is the derivative of the angular velocity which is the derivative of angular position, the inverse mathematical operation is the [antiderivative](http://en.wikipedia.org/wiki/Antiderivative) (or integral):

        <span class="notranslate">$\mathbf{\theta}(t) = \mathbf{\theta}_0 + \int \mathbf{\omega}(t) dt$$\mathbf{\omega}(t) = \mathbf{\omega}_0 + \int \mathbf{\alpha}(t) dt$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Relationship between linear and angular kinematics

        Consider a particle rotating around a point at a fixed distance `r` (circular motion), as the particle moves along the circle, it travels an arc of length `s`.   
        The angular position of the particle is:

        <span class="notranslate">$\theta = \frac{s}{r}$</span>

        Which is in fact similar to the definition of the angular measure radian:

        <div class='center-align'><figure><img src='http://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Radian_cropped_color.svg/220px-Radian_cropped_color.svg.png' width=200/><figcaption><center><i>Figure. An arc of a circle with the same length as the radius of that circle corresponds to an angle of 1 radian (<a href="https://en.wikipedia.org/wiki/Radian">image from Wikipedia</a>).</i></center></figcaption></figure></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Then, the distance travelled by the particle is the arc length:

        <span class="notranslate">$s = r\theta$</span>

        As the radius is constant, the relation between linear and angular velocity and acceleration is straightfoward:

        <span class="notranslate">$v = \frac{ds}{dt} = r\frac{d\theta}{dt} = r\omega$$a = \frac{dv}{dt} = r\frac{d\omega}{dt} = r\alpha$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

        - Read pages 718-742 of the 15th chapter of the [Ruina and Rudra's book] (http://ruina.tam.cornell.edu/Book/index.html) about circular motion of particle.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the Internet

        - Khan Academy: [Uniform Circular Motion Introduction](https://www.khanacademy.org/science/ap-physics-1/ap-centripetal-force-and-gravitation)
        - [Angular Motion and Torque](https://www.youtube.com/watch?v=jNc2SflUl9U)
        - [Rotational Motion Physics, Basic Introduction, Angular Velocity & Tangential Acceleration](https://www.youtube.com/watch?v=WQ9AH2S8B6Y)  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. A gymnast performs giant circles around the horizontal bar (with the official dimensions for  Artistic Gymnastics) at a constant rate of one circle every 2 s and consider that his center of mass is 1 m distant from the bar. At the lowest point (exactly beneath the bigh bar), the gymnast releases the bar, moves forward, and lands standing on the ground.   
         a) Calculate the angular and lineat velocity of the gymnast's center of mass at the point of release.  
         b) Calculate the horizontal distance travelled by the gymnast's center of mass.  

        2. With the data from Table A1 of Winter (2009) and the convention for the sagital joint angles of the lower limb:   
         a. Calculate and plot the angles of the foot, leg, and thigh segments.   
         b. Calculate and plot the angles of the ankle, knee, and hip joint.   
         c. Calculate and plot the velocities and accelerations for the angles calculated in B.
         d. Compare the ankle angle using the two different conventions described by Winter (2009), that is, defining the foot segment with the MT5 or the TOE marker.  
         e. Knowing that a stride period corresponds to the data between frames 1 and 70 (two subsequents toe-off by the right foot), can you suggest a possible candidate for automatic determination of a stride? Hint: look at the vertical displacement and acceleration of the heel marker.  

         [Clik here for the data from Table A.1 (Winter, 2009)](./../data/WinterTableA1.txt) from [Winter's book student site](http://bcs.wiley.com/he-bcs/Books?action=index&bcsId=5453&itemId=0470398183).  

         Example: load data and plot the markers' positions:
        """
    )
    return


@app.cell
def _(pd, plt):
    # load data file
    # use Pandas just to read data from internet
    data = pd.read_csv('https://raw.githubusercontent.com/BMClab/BMC/master/data/WinterTableA1.txt',
                       sep=' ', header=None, skiprows=2).to_numpy()
    markers = ['RIB CAGE', 'HIP', 'KNEE', 'FIBULA', 'ANKLE', 'HEEL', 'MT5', 'TOE']

    fig = plt.figure(figsize=(10, 6))

    ax = plt.subplot2grid((2,2),(0, 0))
    ax.plot(data[: ,1], data[:, 2::2])
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Horizontal [cm]', fontsize=14)

    ax = plt.subplot2grid((2, 2),(0, 1))
    ax.plot(data[: ,1], data[:, 3::2])
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Vertical [cm]', fontsize=14)

    ax = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    ax.plot(data[:, 2::2], data[:, 3::2])
    ax.set_xlabel('Horizontal [cm]', fontsize=14)
    ax.set_ylabel('Vertical [cm]', fontsize=14)
    plt.suptitle('Table A.1 (Winter, 2009): female, 22 yrs, 55.7 kg, 156 cm, ' \
                 'fast cadence (115 steps/min)', y=1.02, fontsize=14)
    ax.legend(markers, loc="upper right", title='Markers')
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Winter DA (2009) [Biomechanics and motor control of human movement](http://books.google.com.br/books?id=_bFHL08IWfwC). 4th edition. Hoboken, EUA: Wiley.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
