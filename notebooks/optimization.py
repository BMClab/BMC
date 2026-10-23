# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "scipy>=1.14", "sympy>=1.13", "urllib3>=2.2", "ipython"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimization

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A net joint moment tells you what the muscles accomplish together, but often not how they share the work. Optimization adds a stated preference among feasible solutions. We first practice on a box and a one-dimensional function, then see how different preferences produce different muscle forces.

    Read the examples in order, predict each result, and then run the cell. Try the distributed Exercises before opening their worked solutions. Finish with the checkpoint questions and independent Problems; their panels contain final answers only.

    **Challenge 0.** Three muscles can balance one measured elbow moment. Would a unique numerical answer prove that the nervous system used those exact forces?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy, SciPy, SymPy, pandas, Matplotlib, and IPython support the examples. The elbow geometry/force tables are public repository assets loaded by `repo_file`. Solver results depend on their units, bounds, and stated objective; Internet access is needed only if those files are not present locally.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="text-align: right">
    <i>If there occur some changes in nature, the amount of action necessary for this change must be as small as possible.</i>
    <br>Maupertuis (sec XVIII)
    </div>

    **Optimization is the process of finding the best value from possible alternatives with regards to a certain criteria** ([Wikipedia](https://en.wikipedia.org/wiki/Mathematical_optimization)).

    Typically, such best value is the value that maximizes or minimizes the criteria. In this context, to solve a (mathematical) optimization problem is to find the maximum or minimum (which need not be a stationary point when it lies on a boundary) of a function (and we can use maximum or minimum interchangeably because the maximum of a function is the minimum of the negative of that function).<br>
    To solve an optimization problem, we first have to model the problem and define the objective, the variables, and the constraints of the problem. In optimization, these terms are usually defined as:

    1. Objective function (or also, cost, loss, utility, or fitness function): a function describing what we want to optimize.
    2. Design variable(s): variables that will be manipulated to optimize the cost function.
    3. Constraint functions: a set of constraints, equalities or inequalities that constrains the possible solutions to possible values of the design variables (candidate solutions or feasible solutions or feasible set).

    A feasible solution that minimizes (or maximizes) the objective function is called an optimal solution.

    The optimization problem is the calculation of the minimum or maximum values of an objective function over a set of **unknown** possible values of the design variables.<br>
    Even in case of a finite number of possible values of the objective function and design variables (e.g., after discretization and a manual or a grid search), in general the evaluation of the objective function is computationally expensive and should be avoided.<br>
    Of note, even if there is no other option, random search can outperform grid search in some high-dimensional parameter searches; this is not a universal guarantee. See [Bergstra, Bengio (2012)](http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf).

    A typical problem of optimization: [Knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem).

    Read more about that in [Introduction to Optimization](https://neos-guide.org/guide/) from the  [NEOS Guide](http://neos-guide.org/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Some jargon in mathematical optimization

    - **Linear versus nonlinear optimization**: linear optimization refers to when the objective function and the constraints are linear mathematical functions. For linear objectives over a convex feasible set, every local minimum is global. If a finite optimum exists on a polytope, at least one vertex attains it; unbounded or infeasible problems need separate treatment. See [Wikipedia 1](https://en.wikipedia.org/wiki/Linear_programming) and [Wikipedia 2](https://en.wikipedia.org/wiki/Nonlinear_programming).
    - **Constrained versus unconstrained optimization**: unconstrained optimization has no constraints; constrained optimization restricts the feasible set.
    - **Convex optimization**: the field of optimization that deals with finding the minimum of convex functions (or the maximum of concave functions) over a convex constraint set. The convexity of a function facilitates the optimization because a local minimum must be a global minimum and first-order conditions (the first derivatives) are sufficient conditions for finding the optimal solution. Note that although convex optimization is a particular case of nonlinear optimization, it is a relatively simple optimization problem, with robust and mature methods of solution. See [Wikipedia](https://en.wikipedia.org/wiki/Convex_optimization).
    - **Multivariate optimization**: optimization of a function of several variables.
    - **Multimodal optimization**: optimization of a function with several local minima to find the multiple (locally) optimal solutions, as opposed to a single best solution.
    - **Multi-objective optimization**: optimization involving more than one objective function to be optimized simultaneously.
    - **Optimal control**: finding a control law for a given system such that a certain optimality criterion is achieved. See [Wikipedia](https://en.wikipedia.org/wiki/Optimal_control).
    - **Quadratic programming**: optimization of a quadratic function subject to linear constraints. See [Wikipedia](https://en.wikipedia.org/wiki/Quadratic_programming).
    - **Simplex algorithm**: linear optimization algorithm that begins at a starting vertex and moves along the edges of the polytope (the feasible region) until it reaches the vertex of the optimum solution. See [Wikipedia](https://en.wikipedia.org/wiki/Simplex_algorithm).
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** For the box problem, identify the design variable, objective, and geometric constraint before differentiating.

    /// details | Solution to Exercise 1
    The cut size x is the variable. Volume is $V=x(a-2x)^2$. A nondegenerate box requires $0<x<a/2$. Checking stationary points without checking this domain can produce an impossible box.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Maxima and minima

    In mathematics, the maximum and minimum of a function are the largest and smallest values that the function takes at a point either within a neighborhood (local) or on the function entire domain  (global) ([Wikipedia](https://en.wikipedia.org/wiki/Maxima_and_minima)).

    For a function of one variable, if the maximum or minimum of a function is not at the limits of the domain and if at least the first and second derivatives of the function exist, a maximum and minimum can be found as the point where the first derivative of the function is zero. If the second derivative on that point is positive, then it's a minimum, if it is negative, it's a maximum.

    <div class='center-align'><figure aria-describedby="figure-caption-1"><img src='https://github.com/BMClab/BMC/blob/master/images/maxmin.png?raw=1' width=350 alt='minima and maxima of a function'/>  </figure></div>

    <span id="figure-caption-1" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. Maxima and minima of a function of one variable.</span>

    - Note that the requirement that the second derivative on the extremum to be positive for a minimum or negative for a maximum is sufficient but not a necessary condition. For instance, the function $f(x)=x^4$ has an extremum in $x=0$ since $f'(x)=4x^3$ and $f'(0)=0$, but its second derivative at $x=0$ is also zero: $f''(x)=12x^2;\: f''(0)=0$. If the first nonzero derivative after the first is of even order, a positive sign gives a local minimum and a negative sign gives a local maximum; an odd order gives neither: $f''''(0)=24$; the extremum is a minimum.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's now apply optimization to solve a problem with a univariate function.
    """)
    return


@app.cell
def _():
    # import Python libraries
    import numpy as np
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib
    import matplotlib.pyplot as plt
    import sympy as sym
    from sympy.plotting import plot
    import pandas as pd
    from IPython.display import display
    from IPython.core.display import Math

    return Math, display, np, pd, plot, plt, sym


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 1: Maximum volume of a cardboard box

    We want to make a box from a square cardboard with side $a$ such that its volume should be maximum.<br>
    What is the optimal distance where the square cardboard should be cut and folded to make a box with maximum volume?

    <div class='center-align'><figure aria-describedby="figure-caption-2"><img src='https://github.com/BMClab/BMC/blob/master/images/box.png?raw=1' width=450 alt='box optimization'/>  </figure></div>

    <span id="figure-caption-2" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. A box to be made from a cardboard such that its volume should be maximum. Where we should cut?</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the distance where to cut and fold the cardboard is $b$, see figure above, the volume of the box will be:

    $$
    \begin{array}{l l}
    V(b) = b(a-2b)(a-2b) \\
    \\
    V(b) = a^2b - 4ab^2 + 4b^3
    \end{array}
    $$

    In the context of optimization:<br>
    **The expression for $V$ is the cost function, $b$ is the design variable, and the constraint is that feasible values of $b$ are in the interval $]0, \dfrac{a}{2}[$, i.e., $b>0$ and $b<\dfrac{a}{2}$.**

    The first and second derivatives of $V$ w.r.t. $b$ are:

    $$
    \begin{array}{l l}
    \dfrac{\mathrm{d}V}{\mathrm{d}b} = a^2 - 8ab + 12b^2 \\
    \\
    \dfrac{\mathrm{d}^2 V}{\mathrm{d}b^2} = - 8a + 24b
    \end{array}
    $$

    We have to find the values for $b$ where the first derivative of $V$ is zero (the extrema) and then use the expression for the second derivative of $V$ to find whether each of these extrema is a minimum (positive value) or a maximum (negative value).<br>
    Let's use Sympy for that:
    """)
    return


@app.cell
def _(Math, display, sym):
    a, b = sym.symbols('a b')
    V = b*(a - 2*b)*(a - 2*b)
    Vdiff = sym.expand(sym.diff(V, b))
    roots = sym.solve(Vdiff, b)
    display(Math(sym.latex('Roots:') + sym.latex(roots)))
    return V, a, b, roots


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Discarding the solution $b=\dfrac{a}{2}$ (where $V=0$, which is a minimum), $b=\dfrac{a}{6}$ results in the maximum volume.<br>
    We can check that by plotting the volume of the cardboard box for $a=1$ and $b: [0,\:0.5]$:
    """)
    return


@app.cell
def _(Math, V, a, b, display, plot, roots, sym):
    plot(V.subs({a: 1}), (b, 0, .5), xlabel='b', ylabel='V')
    display(Math(sym.latex('V_{a=1}^{max}(b=%s)=%s'
                           %(roots[0].evalf(n=4, subs={a: 1}), V.evalf(n=3, subs={a: 1, b: roots[0]})))))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Note that although the problem above is a case of nonlinear constrained optimization, because the objective function is univariate, well-conditioned and the constraints are linear inequalities, the optimization is simple. Unfortunately, this is seldom the case.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** A square sheet has side a. Show that the useful stationary cut is $a/6$ and distinguish it from $a/2$.

    /// details | Solution to Exercise 2
    $V\prime=(a-2x)(a-6x)$, so stationary values are $a/6$ and $a/2$. The second collapses the base and has zero volume. At $a/6$, $V=2a^3/27$, the maximum inside the feasible interval.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Curve fitting as an optimization problem

    Curve fitting is the process of fitting a model, expressed in terms of a mathematical function, that depends on adjustable parameters to a series of data points and once adjusted, that curve has the best fit to the data points.

    The general approach to the fitting procedure involves the definition of a merit function that measures the agreement between data and model. The model parameters are then adjusted to yield the best-fit parameters as a problem of minimization (an optimization problem, where the merit function is the cost function).

    A classical solution, termed least-squares fitting, is to find the best fit by minimizing the sum of the squared differences between data points and the model function (the sum of squared residuals as the merit function).

    For more on curve fitting see the video below and the notebook [Curve fitting](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/curve_fitting.py).
    """)
    return


@app.cell
def _():
    from IPython.display import YouTubeVideo
    YouTubeVideo('Rxp7o7_RxII', width=480, height=360, rel=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gradient descent

    Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function ([Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)).<br>
    In the gradient descent algorithm, a local minimum of a function is found starting from an initial point and taking steps proportional to the negative of the derivative of the function (gradient) at the current point and we evaluate if the current point is lower than then the previous point until a local minimum in reached (hopefully).

    It follows that, if

    $$
    x_{n+1} = x_n - \gamma \nabla f(x)
    $$

    for $\gamma$ small enough, then $f(x_{n}) \geq f(x_{n+1})$.

    This process is repeated iteratively until the step size (which is proportional to the gradient!) is below a required precision (hopefully the sequence $x_{n}$ converges to the desired local minimum).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 2: Minimum of a function by gradient descent

    From https://en.wikipedia.org/wiki/Gradient_descent:<br>
    Calculate the minimum of $f(x)=x^4-3x^3+2$.
    """)
    return


@app.cell
def _():
    cur_x = 6
    gamma = 0.01
    precision = 1e-05
    step_size = 1
    max_iters = 10000
    iters = 0
    f = lambda x: x ** 4 - 3 * x ** 3 + 2
    _df = lambda x: 4 * x ** 3 - 9 * x ** 2
    while (step_size > precision) & (iters < max_iters):
        prev_x = cur_x
        cur_x = cur_x - gamma * _df(prev_x)
        step_size = abs(cur_x - prev_x)
        iters = iters + 1
    print('True local minimum at {} with function value {}.'.format(9 / 4, f(9 / 4)))
    print('Local minimum by gradient descent at {} with function value {}.'.format(cur_x, f(cur_x)))
    return f, max_iters, precision


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The related nonlinear conjugate-gradient algorithm is implemented in the [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize) function, see [minimize(method='CG')](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html), which deals with minimization of a scalar function of one or more variables using the conjugate gradient algorithm.<br>
    Let's run the example above but using the scipy function:
    """)
    return


@app.cell
def _(f, max_iters, np, precision):
    from scipy.optimize import minimize

    minimize(f, 6, args=(), method='CG', jac=None, tol=precision, callback=None,
             options={'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-08,
                      'maxiter': max_iters, 'disp': True, 'return_all': False})
    return (minimize,)




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** For $f(x)=x^2$, use gradient descent $x_{k+1}=x_k-\gamma f\prime(x_k)$. What range of fixed positive step sizes converges to zero?

    /// details | Solution to Exercise 3
    The update is $x_{k+1}=(1-2\gamma)x_k$. Convergence requires $|1-2\gamma|<1$, hence $0<\gamma<1$. A decreasing objective is not guaranteed for an arbitrary large step.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multivariate optimization

    When there is more than one design variable (the cost function depends on more than one variable), it's a multivariate optimization. The general idea of finding minimum and maximum values where the derivatives are zero still holds for a multivariate function. The second derivative of a multivariate function can be described by the Hessian matrix:

    $$
    \mathbf{H} = \begin{bmatrix}{\dfrac  {\partial ^{2}f}{\partial x_{1}^{2}}}&{\dfrac  {\partial ^{2}f}{\partial x_{1}\,\partial x_{2}}}&\cdots &{\dfrac  {\partial ^{2}f}{\partial x_{1}\,\partial x_{n}}}\\[2.2ex]{\dfrac  {\partial ^{2}f}{\partial x_{2}\,\partial x_{1}}}&{\dfrac  {\partial ^{2}f}{\partial x_{2}^{2}}}&\cdots &{\dfrac  {\partial ^{2}f}{\partial x_{2}\,\partial x_{n}}}\\[2.2ex]\vdots &\vdots &\ddots &\vdots \\[2.2ex]{\dfrac  {\partial ^{2}f}{\partial x_{n}\,\partial x_{1}}}&{\dfrac  {\partial ^{2}f}{\partial x_{n}\,\partial x_{2}}}&\cdots &{\dfrac  {\partial ^{2}f}{\partial x_{n}^{2}}}
    \end{bmatrix}
    $$

    Let's see now a classical problem in biomechanics where optimization is useful and there is more than one design variable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The distribution problem in biomechanics

    Using the inverse dynamics approach in biomechanics, we can determine the net force and torque acting on a joint if we know the external forces on the segments and the kinematics and inertial properties of the segments. But with this approach we are unable to determine the individual muscles forces that  created such torque, as expressed in the following equation:

    $$
    M_{total} = M_1 + M_2 + \dots + M_n = r_1F_1 + r_2F_2 + \dots + r_nF_n
    $$

    where $r_i$ is the moment arm of the force $F_i$ that generates a torque $M_i$, a parcel of the (known) total torque $M_{total}$.

    Even if we know the moment arm of each muscle (e.g., from cadaveric data or from image analysis), the equation above has $n$ unknowns. Because there is more than one muscle that potentially created such torque, there are more unknowns than equations, and the problem is undetermined. So, the problem is how to find how the torque is distributed among the muscles of that joint.

    One solution is to consider that we (biological systems) optimize our effort in order to minimize energy expenditure, stresses on our tissues, fatigue, etc. The principle of least action, stated in the opening of this text, is an allusion that optimization might be ubiquitous in nature. With this rationale, let's solve the distribution problem in biomechanics using optimization and find the minimum force of each muscle necessary to complete a given task.

    The following cost functions have been proposed to solve the distribution problem in biomechanics:

    $$
    \begin{array}{l l}
    \displaystyle\sum_{i=1}^N F_i \quad &\text{e.g., Seireg and Arkivar (1973)}
    \\
    \displaystyle\sum_{i=1}^N F_i^2 \quad &
    \\
    \displaystyle\sum_{i=1}^N \left(\dfrac{F_i}{pcsa_i}\right)^2 \quad &\text{e.g., Crowninshield and Brand (1981)}
    \\
    \displaystyle\sum_{i=1}^N \left(\dfrac{F_i}{M_{max,i}}\right)^3 \quad &\text{e.g., Herzog (1987)}
    \end{array}
    $$

    Where $pcsa_i$ is the physiological cross-sectional area of muscle $i$ and $M_{max,i}$ is the maximum torque muscle $i$ can produce.<br>
    Each muscle force $F_i$ is a design variable and the following constraints must be satisfied:

    $$
    \begin{array}{l l}
    0 \leq F_i \leq F_{max}
    \\
    \displaystyle\sum_{i=1}^N r_i \times F_i = M
    \end{array}
    $$

    Let's apply this concept to solve a distribution problem in biomechanics.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.** Two muscles have moment arms 0.02 and 0.04 m. If either alone must supply 4 Nm, what force is needed? Which does minimizing total force favor?

    /// details | Solution to Exercise 4
    The required forces are 200 and 100 N. Minimizing their sum favors the larger moment arm when its capacity permits. That preference is a modeling choice; it is not a direct observation of recruitment.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Muscle force estimation

    Consider the following main flexors of the elbow joint (see figure below): biceps long head, biceps short head, and brachialis. Suppose that the elbow net joint torque determined using inverse dynamics is 20 Nm (flexor). How much each of these muscles contributed to the net torque?

    <div class='center-align'><figure aria-describedby="figure-caption-3"><img src='https://github.com/BMClab/BMC/blob/master/images/elbowflexors.png?raw=1' alt='Elbow flexors'/>  </figure></div>

    <span id="figure-caption-3" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. A view in OpenSim of the arm26 model showing three elbow flexors (Biceps long and short heads and Brachialis).</span>

    For the optimization, we will need experimental data for the moment arm, maximum moment, and *pcsa* of each muscle. Let's import these data from the OpenSim arm26 model:
    """)
    return


@app.function
def repo_file(path, *, base_url="https://raw.githubusercontent.com/BMClab/BMC/master/"):
    """Find a repository file beside this notebook, or fetch a temporary copy.

    Local files take precedence. Only file acquisition falls back to GitHub;
    parsing errors remain visible to the calling notebook. urllib3 selects its
    browser transport automatically without platform-specific imports.
    """
    import hashlib
    import tempfile
    from pathlib import Path, PurePosixPath
    from urllib.parse import quote

    import marimo as mo
    import urllib3

    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(path):
        raise ValueError("Use a repository-relative path such as data/walk.trc")
    location = mo.notebook_dir()
    if location is not None:
        directory = Path(location).resolve()
        for base in (directory, *directory.parents):
            local = base.joinpath(*relative.parts)
            if local.is_file():
                return local
    url = base_url.rstrip("/") + "/" + quote(relative.as_posix(), safe="/")
    key = hashlib.sha256(base_url.rstrip("/").encode()).hexdigest()[:16]
    cached = (Path(tempfile.gettempdir()) / "bmc-notebooks-v2" / key).joinpath(*relative.parts)
    if not cached.is_file():
        response = urllib3.request("GET", url, timeout=30.0, retries=2)
        if response.status != 200:
            raise OSError(f"Cannot load {path}: HTTP {response.status} from {url}")
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as download:
            temporary = Path(download.name)
            download.write(response.data)
        try:
            temporary.replace(cached)
        finally:
            temporary.unlink(missing_ok=True)
    return cached


@app.cell
def _(np):
    # time elbow_flexion BIClong BICshort BRA
    r_ef = np.loadtxt(repo_file('data/r_elbowflexors.mot'), skiprows=7)
    f_ef = np.loadtxt(repo_file('data/f_elbowflexors.mot'), skiprows=7)
    return f_ef, r_ef


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The maximum isometric force of these muscles are defined in the arm26 model as: Biceps long head: 624.3 N, Biceps short head: 435.56 N, and Brachialis: 987.26 N. Let's compute the mamimum torques that each muscle could produce considering a static situation at the different elbow flexion angles:
    """)
    return


@app.cell
def _(f_ef, r_ef):
    m_ef = r_ef*1
    m_ef[:, 2:] = r_ef[:, 2:]*f_ef[:, 2:]
    return (m_ef,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And let's visualize these data:
    """)
    return


@app.cell
def _(f_ef, m_ef, np, plt, r_ef):
    labels = ['Biceps long head', 'Biceps short head', 'Brachialis']
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(10, 4))
    ax[0].plot(r_ef[:, 1], r_ef[:, 2:])
    #ax[0].set_xlabel('Elbow angle$(\,^o)$')
    ax[0].set_title('Moment arm (m)')
    ax[1].plot(f_ef[:, 1], f_ef[:, 2:])
    ax[1].set_xlabel(r'Elbow angle$(\,^o)$', fontsize=16)
    ax[1].set_title('Maximum force (N)')
    ax[2].plot(m_ef[:, 1], m_ef[:, 2:])
    #ax[2].set_xlabel('Elbow angle$(\,^o)$')
    ax[2].set_title('Maximum torque (Nm)')
    ax[2].legend(labels, loc='best', framealpha=.5)
    ax[2].set_xlim(np.min(r_ef[:, 1]), np.max(r_ef[:, 1]))
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These data don't have the *pcsa* value of each muscle. We will estimate the *pcsa* considering that the amount of maximum muscle force generated per area is constant and equal to 50N/cm$^2$. Consequently, the *pcsa* (in cm$^2$) for each muscle is:
    """)
    return


@app.cell
def _(np):
    a_ef = np.array([624.3, 435.56, 987.26])/50  # 50 N/cm2
    print(a_ef)
    return (a_ef,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Static versus dynamic optimization

    In the context of biomechanics, we can solve the distribution problem separately for each angle (instant) of the elbow; we will refer to that as static optimization. However, there is no guarantee that when we analyze all these solutions across the range of angles, they will be the best solution overall. One reason is that static optimization ignores the time history of the muscle force. Dynamic optimization refers to the optimization over a period of time. For such, we will need to input a cost function spanning the entire period of time at once. Dynamic optimization usually has a higher computational cost than static optimization. The term 'Dynamic optimization' used in Biomechanics is not related to the broader method known as '[dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming)' in mathematical optimization or computer programming.

    <div class='center-align'><figure aria-describedby="figure-caption-4"><img src='https://github.com/BMClab/BMC/blob/master/images/optim_stat_dyn.png?raw=1' width=550 alt='static versus dynamic optimization' vspace="10"/>  </figure></div>

    <span id="figure-caption-4" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. Static and dynamic optimization approaches in Biomechanics. Figure taken from Robertson et al. (2013).</span>

    The title 'Static and dynamic optimization solutions for gait are practically equivalent' of an article published by Anderson and Pandy (2001), where they studied normal walking (full gait cycle equals to 1.12 s), summarizes the results of such comparison. The authors reported that the dynamic solution required 1000 times more computation time than the static solution.

    For now, we will solve the present problem using static optimization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Solution of the optimization problem

    For the present case, we are dealing with a problem of minimization, multidimensional (function of several variables), nonlinear, constrained, with linear force bounds and moment balance. For positive normalization constants, the displayed linear, squared, and nonnegative cubic objectives are convex on this feasible set. Numerical optimization is hardly a simple task. There are many different algorithms and public and commercial software for performing optimization. For instance, look at [NEOS Server](http://www.neos-server.org/neos/), a free internet-based service for solving numerical optimization problems.<br>
    We will solve the present problem using the [scipy.optimize](http://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize) package which provides several optimization algorithms. We will use the function `minimize`:

    ```python
    scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
    \"\"\"Minimization of scalar function of one or more variables.\"\"\"
    ```

    Now, let's write Python functions for each cost function:
    """)
    return


@app.cell
def _():
    def cf_f1(x):
        """Cost function: sum of forces."""  
        return x[0] + x[1] + x[2]

    def cf_f2(x):
        """Cost function: sum of forces squared."""
        return x[0]**2 + x[1]**2 + x[2]**2

    def cf_fpcsa2(x, a):
        """Cost function: sum of squared muscle stresses."""
        return (x[0]/a[0])**2 + (x[1]/a[1])**2 + (x[2]/a[2])**2

    def cf_fmmax3(x, m):
        """Cost function: sum of cubic forces normalized by moments."""
        return (x[0]/m[0])**3 + (x[1]/m[1])**3 + (x[2]/m[2])**3

    return cf_f1, cf_f2, cf_fmmax3, cf_fpcsa2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's also define the Jacobian for each cost function (which is an optional parameter for the optimization):
    """)
    return


@app.cell
def _(np):
    def cf_f1d(x):
        """Derivative of cost function: sum of forces."""
        dfdx0 = 1.0
        dfdx1 = 1.0
        dfdx2 = 1.0
        return np.array([dfdx0, dfdx1, dfdx2])

    def cf_f2d(x):
        """Derivative of cost function: sum of forces squared."""
        dfdx0 = 2*x[0]
        dfdx1 = 2*x[1]
        dfdx2 = 2*x[2]
        return np.array([dfdx0, dfdx1, dfdx2])

    def cf_fpcsa2d(x, a):
        """Derivative of cost function: sum of squared muscle stresses."""
        dfdx0 = 2*x[0]/a[0]**2
        dfdx1 = 2*x[1]/a[1]**2
        dfdx2 = 2*x[2]/a[2]**2
        return np.array([dfdx0, dfdx1, dfdx2])

    def cf_fmmax3d(x, m):
        """Derivative of cost function: sum of cubic forces normalized by moments."""
        dfdx0 = 3*x[0]**2/m[0]**3
        dfdx1 = 3*x[1]**2/m[1]**3
        dfdx2 = 3*x[2]**2/m[2]**3
        return np.array([dfdx0, dfdx1, dfdx2])

    return cf_f1d, cf_f2d, cf_fmmax3d, cf_fpcsa2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's define initial values:
    """)
    return


@app.cell
def _(a_ef, f_ef, m_ef, np, r_ef):
    M = 20
    iang = 69
    r = r_ef[iang, 2:]
    f0 = f_ef[iang, 2:]
    a_1 = a_ef
    m = m_ef[iang, 2:]
    x0 = f_ef[iang, 2:] / 10
    print('M =', M)
    print('x0 =', x0)
    print('r * x0 =', np.sum(r * x0))
    return M, a_1, f0, m, r, x0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Inequality constraints (such as boundaries in our problem) can be entered with the parameter `bounds` to the `minimize` function:
    """)
    return


@app.cell
def _(f0):
    bnds = ((0, f0[0]), (0, f0[1]), (0, f0[2]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Equality constraints (such as the sum of torques should equals the desired torque in our problem), as well as inequality constraints, can be entered with the parameter `constraints` to the `minimize` function (and we can also opt to enter the Jacobian of these constraints):
    """)
    return


@app.cell
def _(M, f0, np, r):
    # use this in combination with the parameter bounds:
    cons = ({'type': 'eq',
             'fun' : lambda x, r, f0, M: np.array([r[0]*x[0] + r[1]*x[1] + r[2]*x[2] - M]), 
             'jac' : lambda x, r, f0, M: np.array([r[0], r[1], r[2]]), 'args': (r, f0, M)})
    return


@app.cell
def _(M, f0, np, r):
    cons_1 = ({'type': 'eq', 'fun': lambda x, r, f0, M: np.array([r[0] * x[0] + r[1] * x[1] + r[2] * x[2] - M]), 'jac': lambda x, r, f0, M: np.array([r[0], r[1], r[2]]), 'args': (r, f0, M)}, {'type': 'ineq', 'fun': lambda x, r, f0, M: f0[0] - x[0], 'jac': lambda x, r, f0, M: np.array([-1.0, 0, 0]), 'args': (r, f0, M)}, {'type': 'ineq', 'fun': lambda x, r, f0, M: f0[1] - x[1], 'jac': lambda x, r, f0, M: np.array([0, -1.0, 0]), 'args': (r, f0, M)}, {'type': 'ineq', 'fun': lambda x, r, f0, M: f0[2] - x[2], 'jac': lambda x, r, f0, M: np.array([0, 0, -1.0]), 'args': (r, f0, M)}, {'type': 'ineq', 'fun': lambda x, r, f0, M: x[0], 'jac': lambda x, r, f0, M: np.array([1.0, 0, 0]), 'args': (r, f0, M)}, {'type': 'ineq', 'fun': lambda x, r, f0, M: x[1], 'jac': lambda x, r, f0, M: np.array([0, 1.0, 0]), 'args': (r, f0, M)}, {'type': 'ineq', 'fun': lambda x, r, f0, M: x[2], 'jac': lambda x, r, f0, M: np.array([0, 0, 1.0]), 'args': (r, f0, M)})
    return (cons_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Although more verbose, if all the Jacobians of the constraints are also informed, this alternative seems better than informing bounds for the optimization process (less error in the final result and less iterations).

    We use SLSQP (Sequential Least Squares Programming), which supports equality constraints and bounds. It is one choice, not the only constrained solver in `minimize`; `trust-constr` is another.

    Finally, let's run the optimization for the four different cost functions and find the optimal muscle forces:
    """)
    return


@app.cell
def _(cf_f1, cf_f1d, cons_1, minimize, x0):
    f1r = minimize(fun=cf_f1, x0=x0, args=(), jac=cf_f1d, constraints=cons_1, method='SLSQP', options={'disp': True})
    return (f1r,)


@app.cell
def _(cf_f2, cf_f2d, cons_1, minimize, x0):
    f2r = minimize(fun=cf_f2, x0=x0, args=(), jac=cf_f2d, constraints=cons_1, method='SLSQP', options={'disp': True})
    return (f2r,)


@app.cell
def _(a_1, cf_fpcsa2, cf_fpcsa2d, cons_1, minimize, x0):
    fpcsa2r = minimize(fun=cf_fpcsa2, x0=x0, args=(a_1,), jac=cf_fpcsa2d, constraints=cons_1, method='SLSQP', options={'disp': True})
    return (fpcsa2r,)


@app.cell
def _(cf_fmmax3, cf_fmmax3d, cons_1, m, minimize, x0):
    fmmax3r = minimize(fun=cf_fmmax3, x0=x0, args=(m,), jac=cf_fmmax3d, constraints=cons_1, method='SLSQP', options={'disp': True})
    return (fmmax3r,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's compare the results for the different cost functions:
    """)
    return


@app.cell
def _(M, a_1, f0, f1r, f2r, fmmax3r, fpcsa2r, m, np, pd, r):
    dat = np.vstack((np.around(r * 100, 1), np.around(a_1, 1), np.around(f0, 0), np.around(m, 1)))
    opt = np.around(np.vstack((f1r.x, f2r.x, fpcsa2r.x, fmmax3r.x)), 1)
    er = ['-', '-', '-', '-', np.sum(r * f1r.x) - M, np.sum(r * f2r.x) - M, np.sum(r * fpcsa2r.x) - M, np.sum(r * fmmax3r.x) - M]
    data = np.vstack((np.vstack((dat, opt)).T, er)).T
    rows = ['$\text{Moment arm}\\;[cm]$', '$pcsa\\;[cm^2]$', '$F_{max}\\;[N]$', '$M_{max}\\;[Nm]$', '$\\sum F_i$', '$\\sum F_i^2$', '$\\sum(F_i/pcsa_i)^2$', '$\\sum(F_i/M_{max,i})^3$']
    cols = ['Biceps long head', 'Biceps short head', 'Brachialis', 'Error in M']
    _df = pd.DataFrame(data, index=rows, columns=cols)
    print('\nComparison of different cost functions for solving the distribution problem')
    _df
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check feasibility before interpretation

    After optimization, inspect `success`, the termination message, the moment residual, and every force bound. A returned array is not by itself a feasible solution. Compare starts and solver tolerances, and keep the objective’s units explicit. The retained cubic force-over-maximum-moment criterion has dimensional normalization; it should not be mistaken for dimensionless muscle activation.

    A minimum of a chosen cost is a prediction conditional on that cost. The historical least-action quotation is motivation, not evidence that the nervous system minimizes any particular recruitment objective.
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comments

    The results show that the estimations for the muscle forces depend on the cost function used in the optimization. Which one is correct? This is a difficult question and it's dependent on the goal of the actual task being modeled. Glitsch and Baumann (1997) investigated the effect of different cost functions on the optimization of walking and running and the predicted muscles forces were compared with the electromyographic activity of the corresponding muscles of the lower limb. They found that, among the analyzed cost functions, the minimization of the sum of squared muscle stresses resulted in the best similarity with the actual electromyographic activity.

    In general, one should always test different algorithms and different initial values before settling for the solution found. Downey (2011), Kitchin (2013), and Kiusalaas (2013) present more examples on numerical optimization. The [NEOS Guide](http://neos-guide.org/) is a valuable source of information on this topic and the [SciPy optimization guide](https://docs.scipy.org/doc/scipy/tutorial/optimize.html) documents maintained solver interfaces.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises: apply and audit the models

    **Exercise 5 (original analytical-extrema task).** Find and classify the stationary points of $f(x)=x^3-7.5x^2+18x-10$.

    /// details | Solution to Exercise 5
    The derivative is $3x^2-15x+18=3(x-2)(x-3)$. The second derivative is $6x-15$, negative at 2 and positive at 3. Thus $(2,4)$ is a local maximum and $(3,3.5)$ a local minimum. The cubic is unbounded on the real line, so neither is a global extremum there.
    ///

    **Exercise 6 (original gradient-descent task).** Find the local minimum of the same function by gradient descent, and state a suitable starting point and step.

    /// details | Solution to Exercise 6
    Use $x_{k+1}=x_k-0.05(3x_k^2-15x_k+18)$, starting at $x_0=4$. The iterates approach $x=3$ and $f=3.5$. Starting exactly at 2 stays at a stationary maximum; starting far to its left can run toward the unbounded negative tail. Check the derivative and neighborhood, not only the update size.
    ///

    **Exercise 7 (original elbow-sensitivity task).** Repeat the elbow-force fits from different initial forces and at different elbow angles. Explain what a trustworthy comparison should show.

    /// details | Solution to Exercise 7
    At each angle first load that angle’s moment arms and capacities, then check whether the required moment is feasible. Keep objective and angle fixed while varying initial forces. These convex objectives should converge to the same optimum value when solved accurately; a linear objective can have more than one equally optimal solution. Compare moment residuals and bounds, then vary angle. Changes across angles can reflect changed leverage or active capacity bounds; changes across starts at a fixed angle can expose solver accuracy or nonuniqueness. No single force vector answers every chosen angle.
    ///

    **Exercise 8 (original three-flexor task).** A 10 Nm elbow moment is supplied by brachialis, brachioradialis, and biceps brachii, with maximum forces (1000, 250, 700) N, moment arms (2, 5, 4) cm, and PCSA (33, 8, 23) cm². Estimate forces using different cost functions and test sensitivity to initial values.

    /// details | Solution to Exercise 8
    Convert moment arms to metres and impose $0.02F_1+0.05F_2+0.04F_3=10$ with the stated force bounds. For minimum total force, the largest moment arm can supply the whole moment: $(0,200,0)$ N. For minimum squared force, stationarity gives $F_i=\lambda r_i$, and the constraint sets $\lambda=10/\sum r_i^2$: $(44.444,111.111,88.889)$ N. For squared stress, $F_i=\lambda r_i A_i^2$ and the same constraint gives $(151.040,22.191,146.741)$ N. All satisfy the capacity bounds.

    For the retained cubic force/maximum-moment criterion, use $M_{max,i}=r_iF_{0,i}$: stationarity gives $F_i\propto\sqrt{r_iM_{max,i}^3}$, yielding $(65.459,51.139,153.346)$ N. Repeat with zero, half-capacity, and the feasible vector $(0,200,0)$ as starts; compare the objective and the moment residual, not just printed digits. These calculations show how changing the objective changes the predicted sharing even when the net moment is identical.
    ///
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Is every stationary point a minimum?
    2. What distinguishes a local optimum from a global one?
    3. Why is muscle redundancy underdetermined?
    4. What must be checked after the solver returns?

    /// details | Checkpoint answers
    1. No; it can be a maximum or saddle/inflection point.
    2. The comparison domain: a neighborhood versus the whole feasible set.
    3. There are more unknown muscle forces than independent balance equations.
    4. Status, feasibility, bounds, residuals, and sensitivity to numerical settings.
    ///

    ## Problems

    1. Give the optimal cut and volume for a 30 cm square cardboard.
    2. Give the local maximum and minimum locations of the cubic in Exercise 5.
    3. Two muscles with arms 0.02 and 0.04 m have capacities 50 N each. Can they supply 4 Nm together?

    /// details | Final answers
    1. Cut 5 cm; volume 2000 cm³.
    2. Maximum at x=2; minimum at x=3.
    3. No; their maximum combined moment is 3 Nm.
    ///

    ## Go deeper

    [NEOS optimization guide](https://neos-guide.org/guide/) — classify the objective and constraints before choosing a solver. [SciPy optimization tutorial](https://docs.scipy.org/doc/scipy/tutorial/optimize.html) — current algorithms and result diagnostics.

    ### Video lectures on the Internet

    [Gradient descent, 3Blue1Brown](https://www.youtube.com/watch?v=IHZwWFHWa-w) — visualize a cost landscape and how the gradient directs a local search; the mathematical idea also applies to these mechanical examples.
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Bergstra B, Bengio Y (2012) [Random Search for Hyper-Parameter Optimization](http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf). Journal of Machine Learning Research, 13, 281-305.

    - Crowninshield RD, Brand RA (1981) [A physiologically based criterion of muscle force prediction in locomotion](https://pubmed.ncbi.nlm.nih.gov/7334039). Journal of Biomechanics, 14, 793–801.

    - Downey AB (2014) [Physical Modeling in MATLAB](http://greenteapress.com/wp/physical-modeling-in-matlab-2e/). 2nd edition. Green Tea Press.

    - Herzog W (1987) [Individual muscle force estimations using a non-linear optimal design](https://pubmed.ncbi.nlm.nih.gov/3682873). J Neurosci Methods, 21, 167-179.

    - Glitsch U, Baumann W (1997) [The three-dimensional determination of internal loads in the lower extremity](https://pubmed.ncbi.nlm.nih.gov/9456380). Journal of Biomechanics, 30, 1123–1131.

    - Kitchin J (2013) [pycse - Python Computations in Science and Engineering](http://kitchingroup.cheme.cmu.edu/pycse/).

    - Kiusalaas (2013) [Numerical methods in engineering with Python 3](https://books.google.com.br/books?id=aJkXoxxoCoUC). 3rd edition. Cambridge University Press.

    - Nigg BM and Herzog W (2006) [Biomechanics of the Musculo-skeletal System](https://books.google.com.br/books?id=hOIeAQAAIAAJ&dq=editions:ISBN0470017678). 3rd Edition. Wiley.

    - Robertson G, Caldwell G, Hamill J, Kamen G (2013) [Research Methods in Biomechanics](https://books.google.com.br/books?id=gRn8AAAAQBAJ). 2nd Edition. Human Kinetics.

    - Seireg A, Arvikar RJ (1973) [A mathematical model for evaluation of forces in lower extremeties of the musculo-skeletal system](https://pubmed.ncbi.nlm.nih.gov/4706941). Journal of Biomechanics, 6,  313–322, IN19–IN20, 323–326.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
