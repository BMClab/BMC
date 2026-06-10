import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to numerical solution of Ordinary Differential Equation

    > Marcos Duarte,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this tutorial

    This notebook is written as a guided study session. You will read a short explanation, predict what should happen, run a small computation, and then use the result to answer the next question. Keep a scratchpad nearby and write your answers before you reveal the plots.

    You do not need to finish every challenge on the first pass. Start by running each cell in order. Then return to the challenges and change one parameter at a time. The goal is not only to make the code run; the goal is to connect the equations, the numerical method, and the motion you see in the plots.

    **Challenge 0.** Before you begin, write down one mechanical system you already know that changes over time: a bouncing ball, a pendulum, a person walking, a mass on a spring, or something else. What variable would you measure to describe its state?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## An unassuming derivation of a numerical solution to an Ordinary Differential Equation (ODE)

    Imagine you are in a car. You want to know where the car is at each moment, but you do not have a formula for its position. What you do have is simpler: the initial position of the car and measurements of its speed from time to time.

    That is already enough to make progress.

    Velocity tells us how fast position is changing:

    $$
    v(t) = \frac{\mathrm{d}x(t)}{\mathrm{d}t}.
    $$

    The notation \(x(t)\) and \(v(t)\) is just a reminder that both position and velocity can change with time.

    If we look at two nearby instants, we can approximate the velocity by asking how much the position changed during that time interval:

    $$
    v_i \approx \frac{x_{i+1} - x_i}{t_{i+1} - t_i}.
    $$

    Here, \(i\) labels one measured instant and \(i+1\) labels the next one. To keep the first idea simple, suppose the measurements are equally spaced in time. Then

    $$
    t_{i+1} - t_i = \Delta t,
    $$

    and the approximation becomes

    $$
    v_i \approx \frac{x_{i+1} - x_i}{\Delta t}.
    $$

    Now comes the useful trick. Rearrange this expression to isolate the next position:

    $$
    x_{i+1} \approx x_i + v_i \Delta t.
    $$

    This says something very practical: if you know where the car is now, and you know its current speed, you can estimate where it will be after one small time step.

    For example, suppose

    $$
    x_0 = 100\ \mathrm{m}, \qquad v_0 = 20\ \mathrm{m/s}, \qquad \Delta t = 10\ \mathrm{s}.
    $$

    Then

    $$
    x_1 \approx 100 + 20 \cdot 10 = 300\ \mathrm{m}.
    $$

    The index \(1\) means the next measured instant, which in this example is \(t=10\) s.

    Now suppose that at \(t=10\) s the measured speed is

    $$
    v_1 = 25\ \mathrm{m/s}.
    $$

    Because we already estimated \(x_1\), we can repeat the same step:

    $$
    x_2 \approx x_1 + v_1 \Delta t,
    $$

    so

    $$
    x_2 \approx 300 + 25 \cdot 10 = 550\ \mathrm{m}.
    $$

    Another way to see the same calculation is to keep a running table. The position in each row is the position after applying the speed from the previous interval:

    | \(i\) | Time \(t_i\) [s] | Speed used over previous interval [m/s] | Position \(x_i\) [m] |
    |---:|---:|---:|---:|
    | 0 | 0 | -- | 100 |
    | 1 | 10 | 20 | 300 |
    | 2 | 20 | 25 | 550 |
    | 3 | 30 | 30 | ? |

    The first row is the initial condition. The second row uses \(v_0=20\ \mathrm{m/s}\), and the third row uses \(v_1=25\ \mathrm{m/s}\). Calculate the new position at \(t=30\ \mathrm{s}\).

    And there is the pattern:

    $$
    x_{i+1} \approx x_i + v_i \Delta t.
    $$

    To estimate the position farther into the future, we do not jump there all at once. We move step by step, always using the current position and the current velocity to estimate the next position.

    This is the basic idea behind Euler's method for numerically solving a first-order ordinary differential equation:

    $$
    \frac{\mathrm{d}x(t)}{\mathrm{d}t} = v(t).
    $$

    In this example:

    - \(x_0\) is the initial condition;
    - \(v(t)\) gives the rate of change of position;
    - \(\Delta t\) is the time step;
    - \(x_{i+1} \approx x_i + v_i\Delta t\) is the Euler update.

    Notice what we obtained: a list of estimated positions at selected times. We did not obtain a single formula for \(x(t)\). That is what makes this a numerical solution. Because we started with speed and estimated position by summing increments, this process is numerical integration.
    Because the calculation starts from a known initial value, this kind of problem is called an initial value problem, or IVP.

    **The case when only the acceleration is known**
    Now imagine that we only have access to the car's acceleration, for example from a smartphone accelerometer, and that we know the initial position and speed. Can we still find the position at a later time? Let's try a similar approach for this more challenging problem.

    Acceleration tells us how fast velocity is changing:

    $$
    a(t) = \frac{\mathrm{d}v(t)}{\mathrm{d}t}.
    $$

    If the acceleration is measured at equally spaced instants, we can use the same finite-difference idea:

    $$
    a_i \approx \frac{v_{i+1} - v_i}{\Delta t}.
    $$

    Rearranging gives an update for the next speed:

    $$
    v_{i+1} \approx v_i + a_i\Delta t.
    $$

    Alongside that, we keep using the current speed to update position. So the state we carry from one row to the next is not just position anymore; it is the pair \((x_i, v_i)\). In the simplest Euler version, the next state is defined by these two coupled equations:

    $$
    \left\{
    \begin{array}{l}
    x_{i+1} \approx x_i + v_i\Delta t,\\
    v_{i+1} \approx v_i + a_i\Delta t.
    \end{array}
    \right.
    $$

    They are coupled because the position update needs the current speed, and the speed update needs the current acceleration. This idea will come back later when we rewrite higher-order ODEs as systems of first-order equations.

    Suppose the car starts at \(x_0=100\ \mathrm{m}\), starts with \(v_0=20\ \mathrm{m/s}\), and \(\Delta t=10\ \mathrm{s}\). If the first three acceleration measurements are \(a_0=0.5\ \mathrm{m/s^2}\), \(a_1=0.5\ \mathrm{m/s^2}\), and \(a_2=0.5\ \mathrm{m/s^2}\), the calculation looks like this:

    | \(i\) | Time \(t_i\) [s] | Acceleration used over next interval [m/s^2] | Speed \(v_i\) [m/s] | Position \(x_i\) [m] |
    |---:|---:|---:|---:|---:|
    | 0 | 0 | 0.5 | 20 | 100 |
    | 1 | 10 | 0.5 | 25 | 300 |
    | 2 | 20 | 0.5 | 30 | 550 |
    | 3 | 30 | -- | 35 | ? |

    Use the row at \(t=20\ \mathrm{s}\) to calculate the new position at \(t=30\ \mathrm{s}\). Here, acceleration is first accumulated into speed, and speed is then accumulated into position. This is a double numerical integration: acceleration to velocity, then velocity to position.

    **Challenge 1.** Write a pseudocode to implement the Euler's method for numerically solving a first-order ordinary differential equation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ordinary Differential Equation

    An ODE is an equation that relates a function of one independent variable to one or more of its derivatives. In this tutorial, the independent variable will usually be time. Solving an ODE means finding a function whose derivatives satisfy that relationship.

    The order of an ODE is the order of its highest derivative. For example, a first-order ODE contains only first derivatives. A linear ODE is linear in the unknown function and its derivatives; many important linear ODEs have analytical solutions, but others are still solved numerically in practice. Nonlinear ODEs can sometimes be solved exactly, but numerical methods are often the practical approach.

    An equation is called a partial differential equation when the unknown function depends on more than one independent variable.

    Newton's second law gives a simple example:

    $$
    m\frac{\mathrm{d}^2 \mathbf{x}}{\mathrm{d}t^2}(t) = \mathbf{F}.
    $$

    Here, $\mathbf{x}(t)$ is the position as a function of time and $t$ is the independent variable. The force $\mathbf{F}$ may be constant, such as gravitational force, or it may depend on position, velocity, time, or other quantities. If $\mathbf{F}$ is constant or a linear function of $\mathbf{x}$, this equation is a second-order linear ODE.

    **Guiding questions 1.**
    1. Which quantity is changing with time in Newton's second law?
    2. Which derivative appears in the equation?
    3. What extra information would you need before predicting a unique motion?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First-order ODE

    A first-order ODE has the general form

    $$
    \frac{\mathrm{d} y}{\mathrm{d} x} = f(x, y).
    $$

    The function $f(x, y)$ gives the derivative of $y$ at a given pair $(x, y)$. When $f(x, y)$ is linear with respect to $y$, the equation is a first-order linear ODE, which can be written as

    $$
    \frac{\mathrm{d} y}{\mathrm{d} x} + p(x)y = q(x)
    $$

    where \(p(x)\) and \(q(x)\) are continuous functions of \(x\). In this form, the equation's linearity means the dependent variable \(y\) and its derivative $\mathrm{d}y/\mathrm{d}x$ are raised to the first power and are not multiplied together.

    **Challenge 2.** Suppose $y$ is position and $x$ is time. In one sentence, explain why knowing only $\mathrm{d}y/\mathrm{d}x$ at a single instant is not enough to reconstruct the full motion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial Value Problems (IVP)

    When solving ODEs with a known set of initial conditions, we are solving an initial value problem (IVP). The solution often describes how a system changes over time from a specified initial state.

    A solution to an IVP is a function that satisfies both the differential equation and the initial condition. For a first-order problem,

    $$
    \dot{y}(t) = f(t, y(t)), \qquad y(t_0)=y_0.
    $$

    Higher-order IVPs can be rewritten as systems of first-order IVPs. For example, a second-order equation can be written as

    $$
    \ddot{y}(t) = f(t, y(t), \dot{y}(t)).
    $$

    Define a state vector with the original variable and its derivative:

    $$
    x_1(t) = y(t), \qquad x_2(t) = \dot{y}(t).
    $$

    Then the same problem becomes two coupled first-order equations:

    $$
    \dot{x}_1(t)=x_2(t), \qquad \dot{x}_2(t)=f(t, x_1(t), x_2(t)).
    $$

    In the same way, an ODE of order $N$ can be represented as a system of $N$ first-order ODEs. A simple pendulum, a projectile under gravity, and many biomechanical models are naturally written this way before numerical integration.

    **Guiding question 2.**
    1. What does the initial condition $y(t_0)=y_0$ tell you?

    **Challenge 3.** A mass-spring-damper model can be written as $m \ddot{x} + c \dot{x} + kx = F(t)$. Before looking ahead, define two state variables that would convert this equation into two first-order ODEs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical methods for solving ODEs

    When an ODE is difficult or impossible to solve analytically, numerical methods approximate the solution at selected points. This process is also called numerical integration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Euler method

    The Euler method is the simplest numerical method for solving an initial value problem. First, approximate the derivative of $y$ with a finite difference:

    $$
    \dot{y}(t) \approx \frac{y(t+h)-y(t)}{h},
    $$

    where $h$ is the step size. Rearranging gives

    $$
    y(t+h) \approx y(t) + h\dot{y}(t).
    $$

    Replacing $\dot{y}(t)$ by $f(t, y(t))$ gives

    $$
    y(t+h) \approx y(t) + hf(t, y(t)).
    $$

    Starting from the known value $y(t_0)=y_0$, this becomes the recursive update

    $$
    y_{n+1} = y_n + h f(t_n, y_n).
    $$


    **Guiding questions 3.**
    1. Why do you think numerical integration advances in small time steps?
    2. What might go wrong if the time step is too large?

    **Challenge 4.** Euler's method is simple, but it makes a local straight-line prediction. Sketch what you expect to happen when the real trajectory is strongly curved and the step size $h$ is large.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Euler's method pseudocode

    ```bash
    # Initialization
    Define the differential equation fdot(t, y)
    Input initial time t0, initial value y0
    Input step size h, and the number of steps n
    Create array and store initial values

    # Iteration
    for i from 1 to n:
        # Calculate the slope at the current point
        slope = fdot(t0, y0)
        # Calculate the next time and y-value
        ti = t0 + h
        yi = y0 + h * slope
        # Store new values (ti, yi)
        array[i] = ti, yi
        # Update variables for the next iteration
        t0 = ti
        y0 = yi
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: Exact and approximate solutions for an exponential growth ODE

    Consider $\dot{f(t)}=.8*e^{2*t}$ with initial condition $f_{0}=.4$. In this case, the exact solution is known: $f(t)=.4e^{2*t}$.
    Let's solve this ODE numerically and compare the solutions in the interval $t[0,1]$.

    First, let's import the Python libraries we will use and configure some settings:
    """)
    return


@app.cell
def _():
    import numpy as np

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["lines.linewidth"] = 3
    matplotlib.rcParams["font.size"] = 13
    matplotlib.rcParams["lines.markersize"] = 5
    matplotlib.rc("axes", grid=False, labelsize=14, titlesize=16, ymargin=0.05)
    matplotlib.rc("legend", numpoints=1, fontsize=11)
    return np, plt


@app.cell
def _(np, plt):
    def exp_growth(step=0.1):
        def fdot(t, y):
            return 0.8 * np.exp(2 * t)

        t0, y0 = 0.0, 0.4
        h = step
        n = int(1 / h + 1)
        array = np.empty(shape=[n, 2])
        array[0, :] = t0, y0

        for i in range(1, n):
            slope = fdot(t0, y0)
            ti = t0 + h
            yi = y0 + h * slope
            array[i, :] = ti, yi
            t0, y0 = ti, yi

        t, f_num = array.T
        f_lit = 0.4 * np.exp(2 * t)
        plt.figure()
        plt.plot(t, f_num, "bo:", label="Approximate")
        plt.plot(t, f_lit, "r", label="Exact")
        plt.legend()
        plt.show()

    exp_growth()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Guiding questions 4**
    1. Within the iteration loop, which rows can be reordered and which cannot?
    2. Why pre-assign an array?

    **Challenge 5** Implement this solution without using a pre-assigned array.

    **Challenge 6** Modify the code to increase the step and compare the exact and approximate solutions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explicit and Semi-Implicit Euler Methods

    For a state with position $y$ and velocity $v$, the update order matters. The forward, or explicit, Euler method uses only the current state:

    $$
    y_{i+1}=y_i+h v_i, \qquad v_{i+1}=v_i+h a(t_i, y_i, v_i).
    $$

    A common alternative in mechanics is the semi-implicit Euler method, also called symplectic Euler or Euler-Cromer. It updates velocity first and then uses the new velocity to update position:

    $$
    v_{i+1}=v_i+h a(t_i, y_i, v_i), \qquad y_{i+1}=y_i+h v_{i+1}.
    $$

    This is not the fully implicit backward Euler method, which would evaluate the derivative at the future state and usually requires solving an algebraic equation at every step. The semi-implicit version is still simple to implement, and for many mechanical systems it behaves better than forward Euler.

    **Guiding questions 6.**
    1. In forward Euler, which value of velocity updates position?
    2. In semi-implicit Euler, which value of velocity updates position?
    3. Why might the second choice matter for oscillating systems?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Other numerical methods for solving ODEs

    More accurate methods use additional evaluations of the derivative inside each interval $[t_n, t_{n+1}]$. A common family is the [Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods).

    In SciPy, the current general-purpose interface for initial value problems is [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html). SciPy recommends `solve_ivp` for new code. It provides explicit Runge-Kutta methods such as `RK45` and `DOP853`, implicit methods for stiff systems such as `Radau` and `BDF`, and `LSODA`, which automatically switches between nonstiff and stiff methods.

    **Challenge 7.** When you later compare Euler with `solve_ivp`, predict which method will be more accurate for the same output time step and explain why.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Examples

    Now you will work through examples. Each example starts with a model, rewrites it as an IVP, and then asks you to inspect or modify the numerical solution.

    First, import the required Python libraries and customize the plotting environment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: Simple Pendulum under Gravity

    A simple pendulum is a compact example of an IVP. Let $\theta$ be the angular position and $\omega=\dot{\theta}$ be the angular velocity. For a pendulum of length $d$ under gravity $g$, the equation of motion is

    $$
    \ddot{\theta}(t) = -\frac{g}{d}\sin(\theta(t)).
    $$

    With initial conditions $\theta(0)=45^\circ$ and $\omega(0)=0$, the second-order IVP can be written as two first-order equations:

    $$
    \dot{\theta} = \omega, \qquad
    \dot{\omega} = -\frac{g}{d}\sin(\theta).
    $$

    We will solve this system with the semi-implicit Euler method.

    **Before you run the next cell**, predict whether the angular velocity will be largest near the top of the swing or near the bottom. Then compare your prediction with the plot.
    """)
    return


@app.function
def euler_method(acceleration, T=10, y0=(0.0, 0.0), h=0.01, method=2):
    """
    First-order numerical approximation for two coupled first-order ODEs.

    The state is [position, velocity]. The acceleration function has the
    signature acceleration (t, state) and returns dv/dt.

    The Euler method can be 1 (explicit), or 2 (semi-implicit).
    """
    import numpy as np

    n_samples = int(np.ceil(T / h))
    y = np.zeros((2, n_samples))
    y[:, 0] = np.asarray(y0, dtype=float)
    t = h * np.arange(n_samples)

    for i in range(n_samples - 1):
        if method == 1:
            y[0, i + 1] = y[0, i] + h * y[1, i]
            y[1, i + 1] = y[1, i] + h * acceleration(t[i], y[:, i])
        elif method == 2:
            y[1, i + 1] = y[1, i] + h * acceleration(t[i], y[:, i])
            y[0, i + 1] = y[0, i] + h * y[1, i + 1]
        else:
            raise ValueError("Valid options for method are 1 or 2.")

    return t, y


@app.function
def pendulum_acceleration(t, state):
    """Return angular acceleration for a simple pendulum."""
    import numpy as np

    del t
    length = 0.5
    gravity = 9.8
    return -(gravity / length) * np.sin(state[0])


@app.function
def plot_pendulum(t, y, labels):
    """Plot angular position and velocity for the pendulum example."""
    import matplotlib.pyplot as plt

    _, ax1 = plt.subplots(1, 1, figsize=(10, 4))
    ax1.set_title(labels[0])
    ax1.plot(t, y[0, :], "b", label="Angular position")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(labels[1], color="b")
    ax1.tick_params("y", colors="b")

    ax2 = ax1.twinx()
    ax2.plot(t, y[1, :], "r-.", label="Angular velocity")
    ax2.set_ylabel(labels[2], color="r")
    ax2.tick_params("y", colors="r")
    plt.tight_layout()
    plt.show()


@app.cell
def _(np):
    T_pendulum = 10
    y0_pendulum = [45 * np.pi / 180, 0]
    h_pendulum = 0.01
    t_pendulum, theta = euler_method(
        pendulum_acceleration,
        T=T_pendulum,
        y0=y0_pendulum,
        h=h_pendulum,
        method=2,
    )
    pendulum_labels = [
        "Trajectory of a simple pendulum under gravity",
        "Angular position [deg]",
        "Angular velocity [deg/s]",
    ]
    return pendulum_labels, t_pendulum, theta


@app.cell
def _(np, pendulum_labels, t_pendulum, theta):
    plot_pendulum(t_pendulum, np.rad2deg(theta), pendulum_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Challenge 8.** Change the initial angle to $10^\circ$, $45^\circ$, and $90^\circ$. Does the plot look like a perfect sine wave in all three cases? What does that tell you about the small-angle approximation?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Motion under Constant Force

    Consider a football kicked upward from an initial height $y_0$ with initial vertical velocity $v_0$. We want the equation of motion in the vertical direction.

    Neglecting air resistance, Newton's second law gives

    $$
    m\frac{\mathrm{d}^2 y}{\mathrm{d}t^2} = -mg.
    $$

    We will use $g=9.8\,\mathrm{m/s^2}$, $y_0=1\,\mathrm{m}$ at $t_0=0$, and $v_0=20\,\mathrm{m/s}$. The analytical solution is

    $$
    y(t) = y_0 + v_0 t - \frac{g}{2}t^2.
    $$

    To solve the same problem numerically, rewrite the second-order ODE as two first-order ODEs:

    $$
    \dot{y} = v, \qquad \dot{v} = a.
    $$

    For constant gravitational acceleration,

    $$
    \left\{
    \begin{array}{rl}
    \frac{\mathrm{d} y}{\mathrm{d}t} &= v, \quad y(t_0) = y_0,\\
    \frac{\mathrm{d} v}{\mathrm{d}t} &= -g, \quad v(t_0) = v_0.
    \end{array}
    \right.
    $$

    **Guiding questions 5.**
    1. Which two variables define the state of the ball?
    2. Which equation updates position?
    3. Which equation updates velocity?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This function calculates the vertical trajectory with the Euler method.
    """)
    return


@app.function
def ball_euler(t0, tend, y0, v0, h):
    """Integrate vertical ball motion with the explicit Euler method."""
    import numpy as np

    t = [t0]
    y = [y0]
    v = [v0]
    a = -9.8
    while t[-1] <= tend and y[-1] > 0:
        y.append(y[-1] + h * v[-1])
        v.append(v[-1] + h * a)
        t.append(t[-1] + h)
    return np.array(t), np.array(y), np.array(v)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Set the initial conditions.
    """)
    return


@app.cell
def _():
    y0 = 1
    v0 = 20
    return v0, y0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now integrate the same motion with two step sizes.
    """)
    return


@app.cell
def _(v0, y0):
    t100, y100, v100 = ball_euler(0, 4, y0, v0, 0.1)
    t10, y10, v10 = ball_euler(0, 4, y0, v0, 0.01)
    return t10, t100, v10, v100, y10, y100


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here are the numerical results compared with the analytical solution.
    """)
    return


@app.cell
def _(plt, v0, y0):
    def plots(t100, y100, v100, t10, y10, v10, title):
        """Plot numerical integration results against the analytical solution."""
        a = -9.8

        _, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 5))

        axs[0, 0].plot(
            t10,
            y0 + v0 * t10 + 0.5 * a * t10**2,
            color=[0, 0, 1, 0.7],
            label="Analytical",
        )
        axs[0, 0].plot(t100, y100, "--", color=[0, 1, 0, 0.7], label="h = 100 ms")
        axs[0, 0].plot(t10, y10, ":", color=[1, 0, 0, 0.7], label="h = 10 ms")

        axs[0, 1].plot(t10, v0 + a * t10, color=[0, 0, 1, 0.5], label="Analytical")
        axs[0, 1].plot(t100, v100, "--", color=[0, 1, 0, 0.7], label="h = 100 ms")
        axs[0, 1].plot(t10, v10, ":", color=[1, 0, 0, 0.7], label="h = 10 ms")

        axs[1, 0].plot(
            t10,
            y0 + v0 * t10 + 0.5 * a * t10**2 - (y0 + v0 * t10 + 0.5 * a * t10**2),
            color=[0, 0, 1, 0.7],
            label="Analytical",
        )
        axs[1, 0].plot(
            t100,
            y100 - (y0 + v0 * t100 + 0.5 * a * t100**2),
            "--",
            color=[0, 1, 0, 0.7],
            label="h = 100 ms",
        )
        axs[1, 0].plot(
            t10,
            y10 - (y0 + v0 * t10 + 0.5 * a * t10**2),
            ":",
            color=[1, 0, 0, 0.7],
            label="h = 10 ms",
        )

        axs[1, 1].plot(
            t10,
            v0 + a * t10 - (v0 + a * t10),
            color=[0, 0, 1, 0.7],
            label="Analytical",
        )
        axs[1, 1].plot(
            t100,
            v100 - (v0 + a * t100),
            "--",
            color=[0, 1, 0, 0.7],
            label="h = 100 ms",
        )
        axs[1, 1].plot(
            t10,
            v10 - (v0 + a * t10),
            ":",
            color=[1, 0, 0, 0.7],
            label="h = 10 ms",
        )

        ylabel = ["y [m]", "v [m/s]", "y error [m]", "v error [m/s]"]
        axs[0, 0].set_xlim(t10[0], t10[-1])
        axs[1, 0].set_xlabel("Time [s]")
        axs[1, 1].set_xlabel("Time [s]")
        axs[0, 1].legend()
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            ax.set_ylabel(ylabel[i])
        plt.suptitle(f"Kinematics of a football - {title} method", y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()

    return (plots,)


@app.cell
def _(plots, t10, t100, v10, v100, y10, y100):
    plots(t100, y100, v100, t10, y10, v10, "Euler")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Challenge 9.** Compare the error curves for $h=100$ ms and $h=10$ ms. Which error changes more when the step size decreases: position or velocity? Explain your answer from the update equations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `solve_ivp` can call the LSODA method for the same problem. LSODA is useful as a robust general solver because it switches automatically between methods for nonstiff and stiff systems.
    """)
    return


@app.function
def ball_constant_force(t, yv):
    y, v = yv
    a = -9.8
    return [v, a]


@app.function
def solve_trajectory(fun, t0, tend, yv0, h, method):
    import numpy as np
    from scipy.integrate import solve_ivp

    t_eval = np.arange(t0, tend + h / 2, h)
    solution = solve_ivp(
        fun=fun,
        t_span=(t0, tend),
        y0=yv0,
        method=method,
        t_eval=t_eval,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y[0], solution.y[1]


@app.cell
def _():
    t100_1, y100_1, v100_1 = solve_trajectory(
        ball_constant_force, 0, 4, [1, 20], 0.1, method="LSODA"
    )
    t10_1, y10_1, v10_1 = solve_trajectory(
        ball_constant_force, 0, 4, [1, 20], 0.01, method="LSODA"
    )
    return t100_1, t10_1, v100_1, v10_1, y100_1, y10_1


@app.cell
def _(plots, t100_1, t10_1, v100_1, v10_1, y100_1, y10_1):
    plots(t100_1, y100_1, v100_1, t10_1, y10_1, v10_1, "LSODA")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `solve_ivp` also offers explicit Runge-Kutta methods. Its default method, `RK45`, is an explicit Dormand-Prince method of order 5(4), similar in spirit to MATLAB's `ode45`.
    """)
    return


@app.cell
def _():
    t100_2, y100_2, v100_2 = solve_trajectory(
        ball_constant_force, 0, 4, [1, 20], 0.1, method="RK45"
    )
    t10_2, y10_2, v10_2 = solve_trajectory(
        ball_constant_force, 0, 4, [1, 20], 0.01, method="RK45"
    )
    return t100_2, t10_2, v100_2, v10_2, y100_2, y10_2


@app.cell
def _(plots, t100_2, t10_2, v100_2, v10_2, y100_2, y10_2):
    plots(t100_2, y100_2, v100_2, t10_2, y10_2, v10_2, "RK45")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Challenge 10.** Compare the Euler, LSODA, and RK45 plots. If you had to choose a method for a new biomechanical simulation, what evidence in these plots would influence your choice?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Motion under varying force

    Now include air resistance in the vertical trajectory of the football. Current IFAB Law 2 specifies that a football must be spherical, have a circumference between 68 cm and 70 cm, and weigh between 410 g and 450 g at the start of the match. The representative values used here, circumference $0.69\,\mathrm{m}$ and mass $0.43\,\mathrm{kg}$, are the midpoints of those ranges.

    Model the magnitude of the drag force as

    $$
    F_d(v) = \frac{1}{2}\rho C_d A v^2,
    $$

    where $\rho=1.22\,\mathrm{kg/m^3}$ is air density, $A=0.0379\,\mathrm{m^2}$ is the ball cross-sectional area, and $C_d=0.25$ is an approximate constant drag coefficient from Bray and Kerwin (2003). Drag acts opposite to velocity, so the vertical equation is

    $$
    m\frac{\mathrm{d}^2 y}{\mathrm{d}t^2}
    =
    -mg - \frac{1}{2}\rho C_d A v|v|.
    $$

    As a first-order system,

    $$
    \left\{
    \begin{array}{rl}
    \frac{\mathrm{d} y}{\mathrm{d}t} &= v, \quad y(t_0) = y_0,\\
    \frac{\mathrm{d} v}{\mathrm{d}t} &=
    -g - \frac{1}{2m}\rho C_d A v|v|, \quad v(t_0) = v_0.
    \end{array}
    \right.
    $$

    We will solve this nonlinear IVP numerically with `solve_ivp`.

    **Before you run the next cell**, predict what air resistance will do to the maximum height and time of flight. Will the ball spend more time rising, more time falling, or both?
    """)
    return


@app.function
def ball_with_drag(t, yv):
    import numpy as np

    y, v = yv
    g = 9.8
    m = 0.43
    rho = 1.22
    cd = 0.25
    area = 0.0379
    a = -g - rho * cd * area * v * np.abs(v) / (2 * m)
    return [v, a]


@app.cell
def _():
    t10_3, y10_3, v10_3 = solve_trajectory(
        ball_with_drag, 0, 4, [1, 20], 0.01, method="LSODA"
    )
    return t10_3, v10_3, y10_3


@app.cell
def _(plt, v0, y0):
    def plots_1(t10, y10, v10):
        """Plot vertical ball motion with and without air resistance."""
        a = -9.8
        _, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 5))
        axs[0, 0].plot(
            t10,
            y0 + v0 * t10 + 0.5 * a * t10**2,
            color=[0, 0, 1, 0.7],
            label="No resistance",
        )
        axs[0, 0].plot(t10, y10, "-", color=[1, 0, 0, 0.7], label="With resistance")
        axs[0, 1].plot(t10, v0 + a * t10, color=[0, 0, 1, 0.7], label="No resistance")
        axs[0, 1].plot(t10, v10, "-", color=[1, 0, 0, 0.7], label="With resistance")
        axs[1, 0].plot(
            t10,
            y0 + v0 * t10 + 0.5 * a * t10**2 - (y0 + v0 * t10 + 0.5 * a * t10**2),
            color=[0, 0, 1, 0.7],
            label="No resistance",
        )
        axs[1, 0].plot(
            t10,
            y10 - (y0 + v0 * t10 + 0.5 * a * t10**2),
            "-",
            color=[1, 0, 0, 0.7],
            label="With resistance",
        )
        axs[1, 1].plot(
            t10,
            v0 + a * t10 - (v0 + a * t10),
            color=[0, 0, 1, 0.7],
            label="No resistance",
        )
        axs[1, 1].plot(
            t10,
            v10 - (v0 + a * t10),
            "-",
            color=[1, 0, 0, 0.7],
            label="With resistance",
        )
        ylabel = ["y [m]", "v [m/s]", "y difference [m]", "v difference [m/s]"]
        axs[1, 0].set_xlabel("Time [s]")
        axs[1, 1].set_xlabel("Time [s]")
        axs[0, 1].legend()
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            ax.set_ylabel(ylabel[i])
        plt.suptitle(
            "Kinematics of a football - effect of air resistance", y=1.02, fontsize=16
        )
        plt.tight_layout()
        plt.show()

    return (plots_1,)


@app.cell
def _(plots_1, t10_3, v10_3, y10_3):
    plots_1(t10_3, y10_3, v10_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: Mass-Spring-Damper System

    Now consider a system made of a mass, a spring, and a damper. This is a basic model for many mechanical and biomechanical systems: a shoe midsole, a vehicle suspension, a vibrating instrument string, or a simplified muscle-tendon element.

    Let $x$ be displacement from equilibrium and $v=\dot{x}$ be velocity. A linear mass-spring-damper model is

    $$
    m\ddot{x} + c\dot{x} + kx = F(t),
    $$

    where $m$ is mass, $c$ is damping, $k$ is spring stiffness, and $F(t)$ is an external force. As a first-order IVP,

    $$
    \dot{x}=v, \qquad
    \dot{v}=\frac{F(t)-cv-kx}{m}.
    $$

    **Challenge 11.** Before running the simulation, decide what should happen when damping increases: should the oscillations grow, stay the same, or decay faster?
    """)
    return


@app.function
def mass_spring_damper_acceleration(
    t,
    state,
    mass=1.0,
    stiffness=25.0,
    damping=1.0,
    external_force=0.0,
):
    """Return acceleration for a linear mass-spring-damper system."""
    del t
    position, velocity = state
    return (external_force - damping * velocity - stiffness * position) / mass


@app.function
def simulate_mass_spring_damper(
    mass=1.0,
    stiffness=25.0,
    damping=1.0,
    external_force=0.0,
    initial_state=(0.1, 0.0),
    T=6.0,
    h=0.01,
):
    def acceleration(t, state):
        return mass_spring_damper_acceleration(
            t,
            state,
            mass=mass,
            stiffness=stiffness,
            damping=damping,
            external_force=external_force,
        )

    return euler_method(acceleration, T=T, y0=initial_state, h=h, method=2)


@app.function
def plot_mass_spring_damper(time, state, title):
    import matplotlib.pyplot as plt

    _, ax1 = plt.subplots(1, 1, figsize=(10, 4))
    ax1.set_title(title)
    ax1.plot(time, state[0], "b", label="Displacement")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Displacement [m]", color="b")
    ax1.tick_params("y", colors="b")

    ax2 = ax1.twinx()
    ax2.plot(time, state[1], "r-.", label="Velocity")
    ax2.set_ylabel("Velocity [m/s]", color="r")
    ax2.tick_params("y", colors="r")
    plt.tight_layout()
    plt.show()


@app.cell
def _():
    mass_spring_time, mass_spring_state = simulate_mass_spring_damper(
        mass=1.0,
        stiffness=25.0,
        damping=1.2,
        initial_state=(0.1, 0.0),
        T=6.0,
        h=0.01,
    )
    return mass_spring_state, mass_spring_time


@app.cell
def _(mass_spring_state, mass_spring_time):
    plot_mass_spring_damper(
        mass_spring_time,
        mass_spring_state,
        "Mass-spring-damper response",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Challenge 12.** Try damping values $c=0$, $c=1.2$, and $c=10$ (with $m=1$ and $k=25$, so the natural frequency is $\omega_n=5\,\mathrm{rad/s}$). Classify each response as undamped, underdamped, or critically damped. Which case oscillates without ever settling? Which returns to equilibrium without oscillating? What changes if you double the stiffness?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the final problems. These questions connect the examples in this notebook with mechanics examples used elsewhere in the course.

    1. In the football-with-drag example, what would terminal velocity mean? Which term in the ODE prevents the downward speed from growing without bound?
    2. In a one-mass spring problem, why is the displacement usually measured from the equilibrium position instead of from the unstretched spring length?
    3. If you had a force-platform record from a vertical jump, which signal would you integrate to estimate velocity? What initial condition would you need?
    4. A two-mass spring-damper system needs more state variables than the one-mass example. What are the minimum state variables if the two masses move only along one horizontal line?
    5. In a simple muscle-tendon model, which elements would behave like springs, which element would behave like a damper, and which element would represent active force production?
    6. When would you trust a coarse Euler solution for teaching or estimation, and when would you switch to `solve_ivp` with tighter tolerances?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practice problems

    Work through these in order if you are studying independently. For each one, first write the IVP on paper, then modify or reuse code from this notebook.

    1. **IVP audit.** Explain in your own words why an IVP needs both a differential equation and an initial condition. Then give one example where changing only the initial condition changes the entire motion.

    2. **Pendulum amplitude.** For the pendulum, change only the initial angular velocity. What changes in the plot and what stays similar? Repeat with initial angles of $10^\circ$, $45^\circ$, and $90^\circ$.

    3. **Projectile with terminal velocity.** A $10$ kg projectile is launched vertically upward with $v_0=50\,\mathrm{m/s}$. Compare the maximum height when drag is neglected with the height when the drag force is $F_D=0.01v|v|$. Does the object approach a terminal velocity on the way down?

    4. **Euler step-size audit.** For the football example, find a step size for Euler's method that makes the position error visually small. How small is small enough for your purpose? Explain your criterion.

    5. **Two-dimensional projectile IVP.** Extend the vertical football model to two dimensions. Use state variables $x$, $y$, $v_x$, and $v_y$. Then add drag forces that oppose each velocity component.

    6. **Design a mass-spring-damper system.** Use mass $m=2$ kg and choose stiffness and damping values so the system returns close to equilibrium quickly but does not oscillate more than once. Report the values you tried and justify your final choice.

    7. **One-mass spring problem.** A $1$ kg block is attached to a spring with stiffness $k=100\,\mathrm{N/m}$. At $t=0$, the block is displaced $0.1$ m from equilibrium and released from rest. Write the IVP, simulate the motion with $c=0$, and then find a damping value that removes visible oscillation.

    8. **Compare solvers on an oscillator.** Compare semi-implicit Euler and `solve_ivp` on the same mass-spring-damper system. Which method changes more when you increase the step size? Which result would you use as a reference solution?

    9. **Two-mass spring-damper system.** Two masses move on a horizontal line and are connected by a spring and damper. Define a state vector, write the first-order ODEs, and predict what happens to the distance between the masses when damping is increased.

    10. **Simple muscle-tendon model.** Treat the tendon as one spring, the muscle fibers as another spring, the viscous tissue as a damper, and the contractile element as an active force. Define a one-dimensional IVP for tendon length and explain what each parameter represents.

    11. **Force-platform record.** A force platform measures vertical ground reaction force during a jump. Given body mass and the force-time curve, write the ODE for the center-of-mass acceleration. What numerical integrations would you perform to estimate takeoff velocity and jump height?

    12. **Your own movement model.** Create your own IVP from a movement you care about. Define the state variables, initial conditions, model parameters, and one parameter you would like to estimate from data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Video lectures
    - [Euler's method](https://www.khanacademy.org/math/differential-equations/first-order-differential-equations/eulers-method-tutorial/v/eulers-method) - Khan Academy.
    - Numerical solutions of Ordinary Differential Equations: [Integrating ODEs](https://youtu.be/QBeNXHrAYns?si=lf5vPgCjHs-8-Bc1), [Euler's method](https://youtu.be/MstPeOTCVzQ?si=_kLTgq0d1oI-fZ9O)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Use these references when you want another explanation or more examples:

    - [SciPy `solve_ivp` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) for solver options, events, tolerances, and examples.
    - [SciPy integration tutorial](https://docs.scipy.org/doc/scipy/tutorial/integrate.html) for a broader overview of numerical integration in Python.
    - [OpenStax University Physics: Damped Oscillations](https://openstax.org/books/university-physics-volume-1/pages/15-5-damped-oscillations) for the physics of damping, driven oscillations, and energy loss.
    - [MIT OpenCourseWare: Damped Harmonic Oscillators](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/pages/unit-ii-second-order-constant-coefficient-linear-equations/damped-harmonic-oscillators/) for a differential-equations treatment of spring-mass-damper systems.
    - [IFAB Law 2: The Ball](https://www.theifab.com/laws/latest/the-ball/) for current football size and mass ranges used in the projectile example.
    - [What is the fastest possible volleyball serve?](https://uio-ccse.github.io/computational-essay-showroom/essays/exampleessays/volleyball/Volleyball.html) for an example about the ball movement during a volleyball serve.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Bray K, Kerwin DG (2003) [Modelling the flight of a soccer ball in a direct free kick](https://people.stfx.ca/smackenz/courses/hk474/labs/jump%20float%20lab/bray%202002%20modelling%20the%20flight%20of%20a%20soccer%20ball%20in%20a%20direct%20free%20kick.pdf). Journal of Sports Sciences, 21, 75-85.
    - Downey AB (2023) [Modeling and Simulation in Python: An Introduction for Scientists and Engineers](https://greenteapress.com/wp/modsimpy/). Green Tea Press.
    - IFAB (2025/26) [Current IFAB Law 2: The Ball](https://www.theifab.com/laws/latest/the-ball/).
    - Kiusalaas J (2013) [Numerical Methods in Engineering with Python 3](https://api.pageplace.de/preview/DT0400.9781139604413_A24435840/preview-9781139604413_A24435840.pdf). 3rd edition. Cambridge University Press.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
