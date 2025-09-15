import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # The minimum jerk hypothesis

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
        <center><div style="background-color:#f2f2f2;border:1px solid black;width:72%;padding:5px 10px 5px 10px;text-align:left;">
        <i>"Whatever its physiological underpinnings, the real strength of the minimum-jerk criterion function, or indeed any other criterion function, is its use as an organizing principle. The use of variational principles is common in physics and engineering. They are not presented as the cause of the behavior they describe but rather as a distillation of its essence."</i> &nbsp; <b>Hogan</b> (1984)</div></center>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Python-setup" data-toc-modified-id="Python-setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Python setup</a></span></li><li><span><a href="#Development" data-toc-modified-id="Development-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Development</a></span></li><li><span><a href="#Finding-the-minimum-jerk-trajectory" data-toc-modified-id="Finding-the-minimum-jerk-trajectory-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Finding the minimum jerk trajectory</a></span></li><li><span><a href="#The-angular-trajectory-of-a-minimum-jerk-trajectory" data-toc-modified-id="The-angular-trajectory-of-a-minimum-jerk-trajectory-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>The angular trajectory of a minimum jerk trajectory</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>
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
    # import necessary libraries and configure environment
    import numpy as np
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display, Math
    from sympy import symbols, Matrix, latex, Eq, collect, solve, diff, simplify, init_printing
    from sympy.core import S
    from sympy.utilities.lambdify import lambdify
    init_printing() 
    sns.set_context('notebook', rc={"lines.linewidth": 2})
    return (
        Eq,
        S,
        collect,
        diff,
        display,
        lambdify,
        np,
        plt,
        simplify,
        solve,
        symbols,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Development

        Hogan and Flash (1984, 1985), based on observations of voluntary movements in primates, suggested that movements are performed (organized) with the smoothest trajectory possible. In this organizing principle, the endpoint trajectory is such that the mean squared-jerk across time of this movement is minimum.   

        Jerk is the derivative of acceleration and the observation of the minimum-jerk trajectory is for the endpoint in the extracorporal coordinates (not for joint angles) and according to Flash and Hogan (1985), the minimum-jerk trajectory of a planar movement is such that minimizes the following objective function:$\begin{array}{rcl}
        C = \frac{1}{2} \displaystyle\int\limits_{t_{i}}^{t_{f}}\;\left[\left(\dfrac{d^{3}x}{dt^{3}}\right)^2+\left(\dfrac{d^{3}y}{dt^{3}}\right)^2\right]\;\mathrm{d}t
        
        \end{array}$Hogan (1984) found that the solution for this objective function is a fifth-order polynomial trajectory (see Shadmehr and Wise (2004) for a simpler proof):$\begin{array}{l l}
        x(t) = a_0+a_1t+a_2t^2+a_3t^3+a_4t^4+a_5t^5 \\
        y(t) = b_0+b_1t+b_2t^2+b_3t^3+b_4t^4+b_5t^5
        
        \end{array}$With the following boundary conditions for$x(t)$and$y(t)$: initial and final positions are$(x_i,y_i)$and$(x_f,y_f)$and initial and final velocities and accelerations are zero.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Finding the minimum jerk trajectory

        Let's employ [Sympy](http://sympy.org/en/index.html) to find the solution for the minimum jerk trajectory using symbolic algebra.  
        The equation for minimum jerk trajectory for x is:
        """
    )
    return


@app.cell
def _(Eq, S, display, symbols):
    # symbolic variables
    x, xi, xf, y, yi, yf, d, t = symbols('x, x_i, x_f, y, y_i, y_f, d, t')
    a0, a1, a2, a3, a4, a5 = symbols('a_0:6')
    x = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    display(Eq(S('x(t)'), x))
    return a0, a1, a2, a3, a4, a5, d, t, x, xf, xi, yf, yi


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Without loss of generality, consider$t_i=0$and let's use$d$for movement duration ($d=t_f$).  
        The system of equations with the boundary conditions for$x$is:
        """
    )
    return


@app.cell
def _(Eq, d, diff, display, t, x, xf, xi):
    # define the system of equations
    s = [Eq(x.subs(t, 0), xi),
         Eq(diff(x, t, 1).subs(t, 0),  0),
         Eq(diff(x, t, 2).subs(t, 0),  0),
         Eq(x.subs(t, d), xf),
         Eq(diff(x, t, 1).subs(t, d),  0),
         Eq(diff(x, t, 2).subs(t, d),  0)]
    [display(si) for si in s];
    return (s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Which gives the following solution:
        """
    )
    return


@app.cell
def _(a0, a1, a2, a3, a4, a5, display, s, solve):
    # algebraically solve the system of equations
    sol = solve(s, [a0, a1, a2, a3, a4, a5])
    display(sol)
    return (sol,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Substituting this solution in the fifth order polynomial trajectory equation, we have the actual displacement trajectories:
        """
    )
    return


@app.cell
def _(Eq, S, collect, display, simplify, sol, x, xf, xi, yf, yi):
    # substitute the equation parameters by the solution
    x2 = x.subs(sol)
    x2 = collect(simplify(x2, ratio=1), xf-xi)
    display(Eq(S('x(t)'), x2))
    y2 = x2.subs([(xi, yi), (xf, yf)])
    display(Eq(S('y(t)'), y2))
    return (x2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And for the velocity, acceleration, and jerk trajectories in x:
        """
    )
    return


@app.cell
def _(Eq, S, display, t, x2):
    # symbolic differentiation
    vx = x2.diff(t, 1)
    display(Eq(S('v_x(t)'), vx))
    ax = x2.diff(t, 2)
    display(Eq(S('a_x(t)'), ax))
    jx = x2.diff(t, 3)
    display(Eq(S('j_x(t)'), jx))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's plot the minimum jerk trajectory for x and its velocity, acceleration, and jerk considering$x_i=0,x_f=1,d=1$:
        """
    )
    return


@app.cell
def _(d, diff, lambdify, np, plt, t, x2, xf, xi):
    x3 = x2.subs([(xi, 0), (xf, 1), (d, 1)])
    xfu = lambdify(t, diff(x3, t, 0), 'numpy')
    vfu = lambdify(t, diff(x3, t, 1), 'numpy')
    afu = lambdify(t, diff(x3, t, 2), 'numpy')
    jfu = lambdify(t, diff(x3, t, 3), 'numpy')
    _ts = np.arange(0, 1.01, 0.01)
    (_fig, _axs) = plt.subplots(1, 4, figsize=(10, 4), sharex=True)
    _axs[0].plot(_ts, xfu(_ts), linewidth=3)
    _axs[0].set_title('Displacement [$\\mathrm{m}$]')
    _axs[1].plot(_ts, vfu(_ts), linewidth=3)
    _axs[1].set_title('Velocity [$\\mathrm{m/s}$]')
    _axs[2].plot(_ts, afu(_ts), linewidth=3)
    _axs[2].set_title('Acceleration [$\\mathrm{m/s^2}$]')
    _axs[3].plot(_ts, jfu(_ts), linewidth=3)
    _axs[3].set_title('Jerk [$\\mathrm{m/s^3}$]')
    for _axi in _axs:
        _axi.set_xlabel('Time [s]', fontsize=12)
        _axi.grid(True)
    _fig.suptitle('Minimum jerk trajectory kinematics', fontsize=16)
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that for the minimum jerk trajectory, initial and final values of both velocity and acceleration are zero, but not for the jerk.  

        Read more about the minimum jerk trajectory hypothesis in the [Shadmehr and Wise's book companion site](https://storage.googleapis.com/wzukusers/user-31382847/documents/5a7253343814f4Iv6Hnt/minimumjerk.pdf) and in [Paul Gribble's website](https://gribblelab.org/teaching/compneuro2012/4_Computational_Motor_Control_Kinematics.html#orgheadline12).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The angular trajectory of a minimum jerk trajectory 

        Let's calculate the resulting angular trajectory given a minimum jerk linear trajectory, supposing it is from a circular motion of an elbow flexion. The length of the forearm is 0.5 m, the movement duration is 1 s, the elbow starts flexed at 90$^o$and the flexes to 180$^o$.

        First, the linear trajectories for this circular motion:
        """
    )
    return


@app.cell
def _(Eq, S, d, diff, display, lambdify, t, x2, xf, xi):
    x3_1 = x2.subs([(xi, 0.5), (xf, 0), (d, 1)])
    y3 = x2.subs([(xi, 0), (xf, 0.5), (d, 1)])
    display(Eq(S('y(t)'), x3_1))
    display(Eq(S('x(t)'), y3))
    xfux = lambdify(t, diff(x3_1, t, 0), 'numpy')
    vfux = lambdify(t, diff(x3_1, t, 1), 'numpy')
    afux = lambdify(t, diff(x3_1, t, 2), 'numpy')
    jfux = lambdify(t, diff(x3_1, t, 3), 'numpy')
    xfuy = lambdify(t, diff(y3, t, 0), 'numpy')
    vfuy = lambdify(t, diff(y3, t, 1), 'numpy')
    afuy = lambdify(t, diff(y3, t, 2), 'numpy')
    jfuy = lambdify(t, diff(y3, t, 3), 'numpy')
    return afux, afuy, jfux, jfuy, vfux, vfuy, x3_1, xfux, xfuy, y3


@app.cell
def _(afux, afuy, jfux, jfuy, np, plt, vfux, vfuy, xfux, xfuy):
    _ts = np.arange(0, 1.01, 0.01)
    (_fig, _axs) = plt.subplots(1, 4, figsize=(10, 4), sharex=True)
    _axs[0].plot(_ts, xfux(_ts), 'b', linewidth=3)
    _axs[0].plot(_ts, xfuy(_ts), 'r', linewidth=3)
    _axs[0].set_title('Displacement [$\\mathrm{m}$]')
    _axs[1].plot(_ts, vfux(_ts), 'b', linewidth=3)
    _axs[1].plot(_ts, vfuy(_ts), 'r', linewidth=3)
    _axs[1].set_title('Velocity [$\\mathrm{m/s}$]')
    _axs[2].plot(_ts, afux(_ts), 'b', linewidth=3)
    _axs[2].plot(_ts, afuy(_ts), 'r', linewidth=3)
    _axs[2].set_title('Acceleration [$\\mathrm{m/s^2}$]')
    _axs[3].plot(_ts, jfux(_ts), 'b', linewidth=3)
    _axs[3].plot(_ts, jfuy(_ts), 'r', linewidth=3)
    _axs[3].set_title('Jerk [$\\mathrm{m/s^3}$]')
    for _axi in _axs:
        _axi.set_xlabel('Time [s]', fontsize=12)
        _axi.grid(True)
    _fig.suptitle('Minimum jerk trajectory kinematics', fontsize=16)
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now, the angular trajectories for this circular motion:
        """
    )
    return


@app.cell
def _(Eq, S, diff, display, lambdify, t, x3_1, y3):
    from sympy import atan2, pi
    ang = atan2(y3, x3_1) * 180 / pi
    display(Eq(S('angle(t)'), ang))
    xang = lambdify(t, diff(ang, t, 0), 'numpy')
    vang = lambdify(t, diff(ang, t, 1), 'numpy')
    aang = lambdify(t, diff(ang, t, 2), 'numpy')
    jang = lambdify(t, diff(ang, t, 3), 'numpy')
    return aang, jang, vang, xang


@app.cell
def _(aang, jang, np, plt, vang, xang):
    _ts = np.arange(0, 1.01, 0.01)
    (_fig, _axs) = plt.subplots(1, 4, figsize=(10, 4), sharex=True)
    _axs[0].plot(_ts, xang(_ts), linewidth=3)
    _axs[0].set_title('Displacement [$\\mathrm{^o}$]')
    _axs[1].plot(_ts, vang(_ts), linewidth=3)
    _axs[1].set_title('Velocity [$\\mathrm{^o/s}$]')
    _axs[2].plot(_ts, aang(_ts), linewidth=3)
    _axs[2].set_title('Acceleration [$\\mathrm{^o/s^2}$]')
    _axs[3].plot(_ts, jang(_ts), linewidth=3)
    _axs[3].set_title('Jerk [$\\mathrm{^o/s^3}$]')
    for _axi in _axs:
        _axi.set_xlabel('Time [s]', fontsize=14)
        _axi.grid(True)
    _fig.suptitle('Minimum jerk trajectory angular kinematics', fontsize=16)
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. What is your opinion on the the minimum jerk hypothesis? Do you think humans control movement based on this principle? (Think about what biomechanical and neurophysiological properties are not considered on this hypothesis.)
        2. Calculate and plot the position, velocity, acceleration, and jerk trajectories for different movement speeds (for example, consider always a displacement of 1 m and movement durations of 0.5, 1, and 2 s).  
        3. For the data in the previous item, calculate the ratio peak speed to average speed. Shadmehr and  Wise (2004) argue that psychophysical experiments show that reaching movements with the hand have this ratio equals to 1.75. Compare with the calculated values.  
        4. Can you propose alternative hypotheses for the control of movement?  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Flash T, Hogan N (1985) [The coordination of arm movements: an experimentally confirmed mathematical model](http://www.jneurosci.org/cgi/reprint/5/7/1688.pdf). Journal of Neuroscience, 5, 1688-1703.   
        - Hogan N (1984) [An organizing principle for a class of voluntary movements](http://www.jneurosci.org/content/4/11/2745.full.pdf). Journal of Neuroscience, 4, 2745-2754.
        - Shadmehr R, Wise S (2004) [The Computational Neurobiology of Reaching and Pointing: A Foundation for Motor Learning](https://books.google.com.br/books?id=fKeImql1s_sC&pg=PP1&ots=WuEHfPo6G4&dq=shadmehr&sig=UjsmHL92SidKEKzgvpe5Qu_9pIs&redir_esc=y#v=onepage&q=shadmehr&f=false). A Bradford Book. [Supplementary documents](https://www.shadmehrlab.org/publications).
        - Zatsiorsky VM (1998) [Kinematics of Human Motion](http://books.google.com.br/books/about/Kinematics_of_Human_Motion.html?id=Pql_xXdbrMcC&redir_esc=y). Champaign, Human Kinetics.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
