import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Ordinary Differential Equation

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
        An ordinary differential equation (ODE) is an equation containing a function of one independent variable and its derivatives.  

        Solve an ODE is finding such a function whose derivatives satisfy the equation. The order of an ODE refers to the order of the derivatives; e.g., a first order ODE has only first derivatives. A linear ODE has only linear terms for the function of one independent variable and in general its solution can be obtained analytically. By contrast, a nonlinear ODE doesn't have an exact analytical solution and it has to be solved by numerical methods. The equation is referred as partial differential equation when contains a function of more than one independent variable and its derivatives.  

        A simple and well known example of ODE is Newton's second law of motion:$m\frac{\mathrm{d}^2 \mathbf{x}}{\mathrm{d}t^2}(t) = \mathbf{F}$$\mathbf{x}$is the function with a derivative and$t$is the independent variable. Note that the force,$\mathbf{F}$, can be constant (e.g., the gravitational force) or a function of position,$\mathbf{F}(\mathbf{x}(t))$, (e.g., the force of a spring) or a function of other quantity. If$\mathbf{F}$is constant or a linear function of$\mathbf{x}$, this equation is a second-order linear ODE. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## First-order ODE

        A first-order ODE has the general form:$\frac{\mathrm{d} y}{\mathrm{d} x} = f(x, y)$Where$f(x, y)$is an expression for the derivative of$y$that can be evaluated given$x$and$y$. When$f(x, y)$is linear w.r.t.$y$, the equation is a first-order linear ODE which can be written in the form:$\frac{\mathrm{d} y}{\mathrm{d} x} + P(x)y = Q(x)$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Numerical methods for solving ODE

        When an ODE can't be solved analytically, usually because it's nonlinear, numerical methods are used, a procedure also referred as numerical integration (Downey, 2011; Kitchin, 2013; Kiusalaas, 2013; [Wikipedia](http://en.wikipedia.org/wiki/Numerical_methods_for_ordinary_differential_equations)). In numerical methods, a first-order differential equation can be solved as an Initial Value Problem (IVP) of the form:$\dot{y}(t) = f(t, y(t)), \quad y(t_0) = y_0$In numerical methods, a higher-order ODE is usually transformed into a system of first-order ODE and then this system is solved using numerical integration. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Euler method

        The most simple method to solve an ODE is using the Euler method.  
        First, the derivative of$y$is approximated by:$\dot{y}(t) \approx \frac{y(t+h)-y(t)}{h}$Where$h$is the step size.  
        Rearranging the equation above:$y(t+h) \approx y(t) +h\dot{y}(t)$And replacing$\dot{y}(t)$:$y(t+h) \approx y(t) +hf(t, y(t))$The ODE then can be solved starting at$t_0$, which has a known value for$y_0$:$y(t+h) \approx y_0 + hf(t_0, y_0)$And using the equation recursively for a sequence of values for$t$$(t_0, t_0+h, t_0+2h, ...)$:$y_{n+1} = y_n + hf(t_n, y_n)$This is the Euler method to solve an ODE with a known initial value. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Other numerical methods for solving ODE

        There are other methods for solving an ODE. One family of methods, usually more accurate, uses more points in the interval$[t_n,t_{n+1}]$and are known as [Runge–Kutta methods](http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method). In the Python ecosystem, Runge–Kutta methods are available using the [`scipy.integrate.ode`](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.integrate.ode.html) library of numeric integrators. The library [`scipy.integrate.odeint`](http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html) has other popular integrator known as `lsoda`, from the FORTRAN library odepack.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Examples

        ### Motion under constant force

        Consider a football ball kicked up from an initial height$y_0$and with initial velocity$v_0$. Determine the equation of motion of the ball in the vertical direction.  

        Neglecting the air resistance, Newton's second law of motion applied to this problem for the instants the ball is in the air gives:$m\frac{\mathrm{d}^2 y}{\mathrm{d}t^2} = -mg$Consider$g=9.8m/s^2$,$y_0(t_0=0)=1m$, and$v_0(t_0=0)=20m/s$.

        We know the analytical solution for this problem:$y(t) = y_0 + v_0 t - \frac{g}{2}t^2$Let's solve this problem numerically and compare the results.

        A second-order ODE can be transformed into two first-order ODE, introducing a new variable:$\dot{y} = v$$\dot{v} = a$And rewriting Newton's second law as a couple of equations:$\left\{
        \begin{array}{r}
        \frac{\mathrm{d} y}{\mathrm{d}t} = &v, \quad y(t_0) = y_0
        \\
        \frac{\mathrm{d} v}{\mathrm{d}t} = &-g, \quad v(t_0) = v_0
        \end{array}
        \right.$First, let's import the necessary Python libraries and customize the environment:
        """
    )
    return


@app.cell
def _():
    import numpy as np
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['lines.linewidth'] = 3
    matplotlib.rcParams['font.size'] = 13
    matplotlib.rcParams['lines.markersize'] = 5
    matplotlib.rc('axes', grid=False, labelsize=14, titlesize=16, ymargin=0.05)
    matplotlib.rc('legend', numpoints=1, fontsize=11)
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is the equation for calculating the ball trajectory given the model and using the Euler method:
        """
    )
    return


@app.cell
def _(np):
    def ball_euler(t0, tend, y0, v0, h):
        (t, y, v, i) = ([t0], [y0], [v0], 0)
        a = -9.8
        while t[-1] <= tend and y[-1] > 0:
            y.append(y[-1] + h * v[-1])
            v.append(v[-1] + h * a)
            i = i + 1
            t.append(i * h)
        return (np.array(t), np.array(y), np.array(v))
    return (ball_euler,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Initial values:
        """
    )
    return


@app.cell
def _():
    y0 = 1
    v0 = 20

    a = -9.8
    return v0, y0


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's call the function with different step sizes:
        """
    )
    return


@app.cell
def _(ball_euler, v0, y0):
    t100, y100, v100 = ball_euler(0, 10, y0, v0, 0.1)
    t10, y10, v10    = ball_euler(0, 10, y0, v0, 0.01)
    return t10, t100, v10, v100, y10, y100


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here are the plots for the results:
        """
    )
    return


@app.cell
def _(plt, v0, y0):
    def plots(t100, y100, v100, t10, y10, v10, title):
        """Plots of numerical integration results.
        """
        a = -9.8
    
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 5))

        axs[0, 0].plot(t10, y0 + v0*t10 + 0.5*a*t10**2, color=[0, 0, 1, .7], label='Analytical')
        axs[0, 0].plot(t100, y100, '--', color=[0, 1, 0, .7], label='h = 100ms')
        axs[0, 0].plot(t10, y10, ':', color=[1, 0, 0, .7], label='h =   10ms')

        axs[0, 1].plot(t10, v0 + a*t10, color=[0, 0, 1, .5], label='Analytical')
        axs[0, 1].plot(t100, v100, '--', color=[0, 1, 0, .7], label='h = 100ms')
        axs[0, 1].plot(t10, v10, ':', color=[1, 0, 0, .7], label='h =   10ms')

        axs[1, 0].plot(t10, y0 + v0*t10 + 0.5*a*t10**2 - (y0 + v0*t10 + 0.5*a*t10**2),
                       color=[0, 0, 1, .7], label='Analytical')
        axs[1, 0].plot(t100, y100 - (y0 + v0*t100 + 0.5*a*t100**2), '--',
                       color=[0, 1, 0, .7], label='h = 100ms')
        axs[1, 0].plot(t10, y10 - (y0 + v0*t10 + 0.5*a*t10**2), ':',
                       color=[1, 0, 0, .7], label='h =   10ms')

        axs[1, 1].plot(t10, v0 + a*t10 - (v0 + a*t10), color=[0, 0, 1, .7], label='Analytical')
        axs[1, 1].plot(t100, v100 - (v0 + a*t100), '--', color=[0, 1, 0, .7], label='h = 100ms')
        axs[1, 1].plot(t10, v10 - (v0 + a*t10), ':', color=[1, 0, 0, .7], label='h =   10ms')

        ylabel = ['y [m]', 'v [m/s]', 'y error [m]', 'v error [m/s]']
        axs[0, 0].set_xlim(t10[0], t10[-1])
        axs[1, 0].set_xlabel('Time [s]')
        axs[1, 1].set_xlabel('Time [s]')
        axs[0, 1].legend()
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            ax.set_ylabel(ylabel[i])
        plt.suptitle('Kinematics of a soccer ball - %s method'%title, y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
    return (plots,)


@app.cell
def _(plots, t10, t100, v10, v100, y10, y100):
    plots(t100, y100, v100, t10, y10, v10, 'Euler')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's use the integrator `lsoda` to solve the same problem:
        """
    )
    return


@app.cell
def _():
    from scipy.integrate import odeint, ode

    def ball_eq(yv, t):
    
        y = yv[0]  # position 
        v = yv[1]  # velocity
        a = -9.8   # acceleration
    
        return [v, a]
    return ball_eq, ode, odeint


@app.cell
def _(ball_eq, np, odeint):
    _yv0 = [1, 20]
    t10_1 = np.arange(0, 4, 0.1)
    _yv10 = odeint(ball_eq, _yv0, t10_1)
    (y10_1, v10_1) = (_yv10[:, 0], _yv10[:, 1])
    t100_1 = np.arange(0, 4, 0.01)
    yv100 = odeint(ball_eq, _yv0, t100_1)
    (y100_1, v100_1) = (yv100[:, 0], yv100[:, 1])
    return t100_1, t10_1, v100_1, v10_1, y100_1, y10_1


@app.cell
def _(plots, t100_1, t10_1, v100_1, v10_1, y100_1, y10_1):
    plots(t100_1, y100_1, v100_1, t10_1, y10_1, v10_1, 'lsoda')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's use an explicit runge-kutta method of order (4)5 due to Dormand and Prince (a.k.a. ode45 in Matlab):
        """
    )
    return


@app.function
def ball_eq_1(t, yv):
    y = yv[0]
    v = yv[1]
    a = -9.8
    return [v, a]


@app.cell
def _(np, ode):
    def ball_sol(fun, t0, tend, yv0, h):
        f = ode(fun).set_integrator('dopri5')
        f.set_initial_value(_yv0, t0)
        data = []
        while f.successful() and f.t < tend:
            f.integrate(f.t + h)
            data.append([f.t, f.y[0], f.y[1]])
        data = np.array(data)
        return data
    return (ball_sol,)


@app.cell
def _(ball_sol):
    data = ball_sol(ball_eq_1, 0, 4, [1, 20], 0.1)
    (t100_2, y100_2, v100_2) = (data[:, 0], data[:, 1], data[:, 2])
    data = ball_sol(ball_eq_1, 0, 4, [1, 20], 0.01)
    (t10_2, y10_2, v10_2) = (data[:, 0], data[:, 1], data[:, 2])
    return t100_2, t10_2, v100_2, v10_2, y100_2, y10_2


@app.cell
def _(plots, t100_2, t10_2, v100_2, v10_2, y100_2, y10_2):
    plots(t100_2, y100_2, v100_2, t10_2, y10_2, v10_2, 'dopri5 (ode45)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Motion under varying force

        Let's consider the air resistance in the calculations for the vertical trajectory of the football ball.  
        According to the Laws of the Game from FIFA, the ball is spherical, has a circumference of$0.69m$, and a mass of$0.43kg$.  
        We will model the magnitude of the [drag force](http://en.wikipedia.org/wiki/Drag_%28physics%29) due to the air resistance by:$F_d(v) = \frac{1}{2}\rho C_d A v^2$Where$\rho$is the air density$(1.22kg/m^3)$,$A$the ball cross sectional area$(0.0379m^2)$, and$C_d$the drag coefficient, which for now we will consider constant and equal to$0.25$(Bray and Kerwin, 2003).  
        Applying Newton's second law of motion to the new problem:$m\frac{\mathrm{d}^2 y}{\mathrm{d}t^2} = -mg -\frac{1}{2}\rho C_d A v^2\frac{v}{||v||}$In the equation above,$-v/||v||$takes into account that the drag force always acts opposite to the direction of motion.  
        Reformulating the second-order ODE above as a couple of first-order equations:$\left\{
        \begin{array}{l l}
        \frac{\mathrm{d} y}{\mathrm{d}t} = &v, \quad &y(t_0) = y_0
        \\
        \frac{\mathrm{d} v}{\mathrm{d}t} = &-g -\frac{1}{2m}\rho C_d A v^2\frac{v}{||v||}, \quad &v(t_0) = v_0
        \end{array}
        \right.$Although (much) more complicated, it's still possible to find an analytical solution for this problem. But for now let's explore the power of numerical integration and use the `lsoda` method (the most simple method to call in terms of number of lines of code) to solve this problem:
        """
    )
    return


@app.cell
def _(np):
    def ball_eq_2(yv, t):
        g = 9.8
        m = 0.43
        rho = 1.22
        cd = 0.25
        A = 0.0379
        y = yv[0]
        v = yv[1]
        a = -g - 1 / (2 * m) * rho * cd * A * v * np.abs(v)
        return [v, a]
    return (ball_eq_2,)


@app.cell
def _(ball_eq_2, np, odeint):
    _yv0 = [1, 20]
    t10_3 = np.arange(0, 4, 0.01)
    _yv10 = odeint(ball_eq_2, _yv0, t10_3)
    (y10_3, v10_3) = (_yv10[:, 0], _yv10[:, 1])
    return t10_3, v10_3, y10_3


@app.cell
def _(plt, v0, y0):
    def plots_1(t10, y10, v10):
        """Plots of numerical integration results.
        """
        a = -9.8
        (fig, axs) = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 5))
        axs[0, 0].plot(t10, y0 + v0 * t10 + 0.5 * a * t10 ** 2, color=[0, 0, 1, 0.7], label='No resistance')
        axs[0, 0].plot(t10, y10, '-', color=[1, 0, 0, 0.7], label='With resistance')
        axs[0, 1].plot(t10, v0 + a * t10, color=[0, 0, 1, 0.7], label='No resistance')
        axs[0, 1].plot(t10, v10, '-', color=[1, 0, 0, 0.7], label='With resistance')
        axs[1, 0].plot(t10, y0 + v0 * t10 + 0.5 * a * t10 ** 2 - (y0 + v0 * t10 + 0.5 * a * t10 ** 2), color=[0, 0, 1, 0.7], label='Real')
        axs[1, 0].plot(t10, y10 - (y0 + v0 * t10 + 0.5 * a * t10 ** 2), '-', color=[1, 0, 0, 0.7], label='h=10 ms')
        axs[1, 1].plot(t10, v0 + a * t10 - (v0 + a * t10), color=[0, 0, 1, 0.7], label='No resistance')
        axs[1, 1].plot(t10, v10 - (v0 + a * t10), '-', color=[1, 0, 0, 0.7], label='With resistance')
        ylabel = ['y [m]', 'v [m/s]', 'y diff [m]', 'v diff [m/s]']
        axs[1, 0].set_xlabel('Time [s]')
        axs[1, 1].set_xlabel('Time [s]')
        axs[0, 1].legend()
        axs = axs.flatten()
        for (i, ax) in enumerate(axs):
            ax.set_ylabel(ylabel[i])
        plt.suptitle('Kinematics of a soccer ball - effect of air resistance', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
    return (plots_1,)


@app.cell
def _(plots_1, t10_3, v10_3, y10_3):
    plots_1(t10_3, y10_3, v10_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exercises

        1. Run the simulations above considering different values for the parameters.  
        2. Model and run simulations for the two-dimensional case of the ball trajectory and investigate the effect of air resistance. Hint: chapter 9 of Downey (2011) presents part of the solution.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Bray K, Kerwin DG (2003) [Modelling the flight of a soccer ball in a direct free kick](http://people.stfx.ca/smackenz/Courses/HK474/Labs/Jump%20Float%20Lab/Bray%202002%20Modelling%20the%20flight%20of%20a%20soccer%20ball%20in%20a%20direct%20free%20kick.pdf). Journal of Sports Sciences, 21, 75–85.   
        - Downey AB (2011) [Physical Modeling in MATLAB](http://greenteapress.com/matlab/). Green Tea Press.  
        - FIFA (2015) [Laws of the Game 2014/2015](http://www.fifa.com/aboutfifa/footballdevelopment/technicalsupport/refereeing/laws-of-the-game/).
        - Kitchin J (2013) [pycse - Python Computations in Science and Engineering](http://kitchingroup.cheme.cmu.edu/pycse/).  
        - Kiusalaas (2013) [Numerical methods in engineering with Python 3](http://books.google.com.br/books?id=aJkXoxxoCoUC). 3rd edition. Cambridge University Press.  
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
