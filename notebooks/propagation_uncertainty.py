# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "sympy>=1.13", "uncertainties>=3.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Propagation of uncertainty (error)

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A calculator reports many digits, but the measurements behind them may be uncertain. We start with density, work through a first-order uncertainty calculation, and let Python repeat the arithmetic and symbolic differentiation. Keep the physical units and the meaning of one standard deviation visible.

    Read the examples in order, predict each result, and then run the cell. Try the distributed Exercises before opening their worked solutions. Finish with the checkpoint questions and independent Problems; their panels contain final answers only.

    **Challenge 0.** A density is mass divided by volume. If both measurements have uncertainty, is reporting their ratio to six decimal places useful? Which measurement would you improve first?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    The `uncertainties` package tracks first-order propagation and shared variables; SymPy derives formulas. Both are declared dependencies. All examples use stated hypothetical measurements, not external files. The notebook fails visibly if a required package is unavailable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation of measurement data

    > The result of a measurement is only an approximation or estimate of the value of the measurand and thus is complete only when accompanied by a statement of the uncertainty of that estimate.
    >
    > Uncertainty (of measurement) is a parameter, associated with the result of a measurement, that characterizes the dispersion of the values that could reasonably be attributed to the measurand.
    >
    > [Evaluation of measurement data - Guide to the expression of uncertainty in measurement (2008)](https://doi.org/10.59161/JCGM100-2008E)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For instance, to calculate the density of a material someone made the following measurements of mass and volume ($d=m/V$):

    $$
    m = 4.0 \pm 0.5 \; g
    $$

    $$
    V = 2.0 \pm 0.2 \; cm^3
    $$

    Where 0.5 g and 0.2 cm$^3$ each represent a value of one standard deviation and express the standard uncertainties of the measured mass and volume.

    Based on these numbers, the material density is 2 g/cm$^3$; but how much is the error (uncertainty) of the density?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One way to estimate the error in the density is by propagation of uncertainty:

    > Propagation of uncertainty (or propagation of error) is the effect of variables' uncertainties (or errors) on the uncertainty of a function based on them. When the variables are the values of experimental measurements they have uncertainties due to measurement limitations (e.g., instrument precision and noise) which propagate to the combination of variables in the function ([Wikipedia](https://en.wikipedia.org/wiki/Propagation_of_uncertainty)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Propagation of uncertainty by linear approximation

    The classic way for propagation of uncertainty is through linear approximation of error propagation theory. We start with the mathematical function that relates the physical quantities and estimate the contribution of their variations by means of partial derivatives (the rate of variation) of each of these primary quantities. The linearity of the mathematical function in the vicinity of the obtained value and the statistical correlations between these quantities are also taken into consideration in the propagation of uncertainty.

    If the measurand $f$ is defined in terms of the variables $x, y, z, ...$ by a general expression $f(x, y, z, ...)$, a first order approximation of the propagation of uncertainty ignoring correlations between these variables is given by the well known formula:

    $$
    \sigma _{f}\;=\;\sqrt{\left ( \frac{\partial f}{\partial x}  \right)^2\sigma _{x}^{2} + \left ( \frac{\partial f}{\partial y} \right)^2 \sigma _{y}^{2} + \left ( \frac{\partial f}{\partial z} \right)^2 \sigma _{z}^{2}\: + \ldots}
    $$

    Where $\partial f/\partial i$ is the first partial derivative of $f$ with respect to variable $i$ and $\sigma_i$ is the uncertainty of the measurement of the variable $i$ (one standard deviation).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using the former formula for linear approximation of error propagation, the estimation of the uncertainty of the density is given by:

    $$
    d=\frac{m}{V} \\
    \sigma _{d}\;=\;\sqrt{ \left ( \frac{\partial d}{\partial m}  \right)^2 \sigma _{m}^{2}+\left ( \frac{\partial d}{\partial V} \right)^2 \sigma _{V}^{2} } \\
    \sigma _{d}\;=\;\sqrt{ \left ( \frac{1}{V} \right)^2 \sigma _{m}^{2}+\left ( -\frac{m}{V^2} \right)^2 \sigma _{V}^{2} }
    $$

    And considering the values of mass and volume given earlier:

    $$
    \sigma_{d}\;=\;\sqrt{\left(\frac{1}{2}\right)^2\times 0.5\:^2+\left(-\frac{4}{2^2}\right)^2\times 0.2\:^2}\;\;\approx\; 0.32
    $$

    Then, the estimated value of the density can now be expressed in a complete form: d = 2.00±0.32 g/cm$^3$.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** Using the density example, compare the mass and volume contributions to the standard uncertainty.

    /// details | Solution to Exercise 1
    The magnitudes are $(1/V)u_m=0.25$ and $(m/V^2)u_V=0.20$ g/cm³. Their squared sum is $0.0625+0.04$, giving $u_d=0.3202$ g/cm³. Improving the mass measurement removes the slightly larger contribution.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using the `uncertainties` package to calculate uncertainty

    It is essential that any result of a measurement is accompanied by the estimate of its uncertainty as above, this way we can better understand the value of the measure and how reliable it is. However, it is unlikely that anyone will repeat these calculations in the day-to-day work (although there are tables with partial derivatives for the most common expressions). Therefore, it is useful a software that can perform these calculations (at least for checking the results manually obtained). Fortunately, there are software for the propagation of uncertainty, [see this list on Wikipedia](https://en.wikipedia.org/wiki/List_of_uncertainty_propagation_software).

    One such software, free, open source, and cross-platform, is the library for [Python](https://python.org) called [uncertainties](https://pypi.org/project/uncertainties/). With [uncertainties](https://pypi.org/project/uncertainties/) installed, let's calculate again the uncertainty of the density:
    """)
    return


@app.cell
def _():
    from uncertainties import ufloat
    return (ufloat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's input the values:
    """)
    return


@app.cell
def _(ufloat):
    _m = ufloat(nominal_value=4.0, std_dev=0.5)
    _V = ufloat(2.0, 0.2)
    d = _m / _V
    print(d)
    return (d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The result we obtained before, but without having to deduct the partial derivatives because [uncertainties](https://uncertainties.readthedocs.io/en/latest/) does this with an [algorithm for automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Displaying the output in different formats
    """)
    return


@app.cell
def _(d, mo):
    mo.output.append(mo.md(f'$$d = {d:.3L} g/cm^3$$'))
    mo.output.append(mo.md(f'$$d = {d:.2eL} g/cm^3$$'))
    mo.output.append(mo.md(f'$$d = {d:.2uS} g/cm^3$$'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The operation above could have been performed in a single line (after you imported the necessary libraries at the first time using them):
    """)
    return


@app.cell
def _(ufloat):
    print(ufloat(4.0, 0.5) / ufloat(2.0, 0.2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And how much each variable mass and volume contributed to the final uncertainty (i.e., the partial derivative of density in relation to the variable mass or volume times its standard deviation (each term on the right side of the equation above without squaring)) is given by:
    """)
    return


@app.cell
def _(d):
    d.error_components().items()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By definition, according to the formula above, the square root of the quadratic sum of these values is the total uncertainty for the density.

    The value of each partial derivative of the density function with respect to the mass and volume variables is given by:
    """)
    return


@app.cell
def _(d):
    d.derivatives
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** If `x` is one uncertain measurement, what should the uncertainty of `x-x` be? What if two independent measurements merely have identical reported values?

    /// details | Solution to Exercise 2
    The same measurement cancels exactly, so `x-x` has zero uncertainty. Two independently created quantities with SD $u$ have difference SD $\sqrt2u$. Equal nominal values do not imply perfect correlation.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Automatic deduction of the symbolic formula for error propagation

    Stepping back, we can even deduce the formula for the error propagation using the partial derivatives in case we want the actual formula using [Sympy](http://sympy.org/en/index.html):
    """)
    return


@app.cell
def _(mo):
    from sympy import symbols, Symbol, sqrt, Add, Mul, Pow, init_printing, latex
    init_printing()

    def stdform(formula, formula_symb, *errors):
        """
        Calculate the symbolic formula for error propagation using partial derivatives.
        """
        from functools import reduce
        formula2 = sqrt(reduce(Add, (Mul(Pow(formula.diff(var), 2, evaluate=False), Symbol(f'\\sigma_{var.name}', commutative=False) ** 2, evaluate=False) for var in formula.atoms(Symbol) if var.name in errors), 0.0))
        formula3 = sqrt(reduce(Add, (Mul(Pow(formula.diff(var) / formula, 2, evaluate=False), Symbol(f'\\sigma_{var.name}', commutative=False) ** 2, evaluate=False) for var in formula.atoms(Symbol) if var.name in errors), 0.0))
        print('Uncertainty:')
        mo.output.append(mo.md(f'$$\\sigma_{formula_symb} = {latex(formula2)}$$'))
        print('Relative Uncertainty:')
        mo.output.append(mo.md(f'$$\\frac{{\\sigma_{formula_symb}}}{formula_symb} = {latex(formula3)}$$'))
        return (formula2, formula3)

    return stdform, symbols


@app.cell
def _(stdform, symbols):
    print('Example for uncertainty propagation for the density, d=m/v:')
    (_m, _V) = symbols('m, V')
    d_1 = _m / _V
    _formula = stdform(d_1, 'd', 'm', 'V')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same as before, but Sympy deduced for us.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Another example

    Let's estimate the uncertainty in the measurement of the volume of a cylinder given its diameter ($d=2.0\pm0.1$) cm and height $h=10.0\pm0.5$ cm. But now, let's go straight to the solution:
    """)
    return


@app.cell
def _(stdform, symbols):
    from sympy import pi as _pi
    (d_2, _h) = symbols('d, h')
    _V = _h * _pi * (d_2 / 2) ** 2
    _formula = stdform(_V, 'V', 'd', 'h')
    return


@app.cell
def _(mo, ufloat):
    from math import pi as _pi
    _h = ufloat(nominal_value=10.0, std_dev=0.5)
    d_3 = ufloat(nominal_value=2.0, std_dev=0.1)
    _V = _h * _pi * (d_3 / 2) ** 2
    mo.output.append(mo.md(f'$$V = {_V:.3L} cm^3$$'))
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** For a cylinder $V=\pi d^2h/4$, what happens to relative volume uncertainty when diameter uncertainty doubles and height is exact?

    /// details | Solution to Exercise 3
    The first-order relation is $u_V/V=2u_d/d$ when height is exact. Doubling $u_d$ doubles $u_V/V$. The square on diameter makes relative diameter uncertainty twice as influential as equal relative height uncertainty.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other Python packages for uncertainty analysis

    The `uncertainties` package propagates means and standard deviations using a local linear approximation. It does not require Gaussian inputs for that propagation, but converting the result into a coverage probability needs distributional assumptions. Strong nonlinearity over the uncertainty range can make the approximation inaccurate. Alternatives include:<br>
     - [NIST Uncertainty Machine](https://uncertainty.nist.gov/): *A Web-based software application to evaluate the measurement uncertainty associated with an output quantity defined by a measurement model of the form $y=f(x_0,	\ldots,x_n)$.*  The NIST Uncertainty Machine allows to select a probability distribution for each of the input quantities and can evaluate the measurement uncertainty using two methods: the linear approximation and the Monte Carlo method,
     - [mcerp](https://github.com/tisimst/mcerp): *A stochastic calculator for Monte Carlo methods that uses latin-hypercube sampling to perform non-order specific error propagation (or uncertainty analysis).*<br>
     - [soerp](https://github.com/tisimst/soerp): *Python implementation of the original Fortran code SOERP by N. D. Cox to apply a second-order analysis to error propagation (or uncertainty analysis). The soerp package allows you to easily and transparently track the effects of uncertainty through mathematical calculations.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And there are other packages using linear error propagation:

     - [scinum](https://github.com/riga/scinum): *Provides a simple Number class that wraps plain floats or NumPy arrays and adds support for multiple uncertainties, automatic (gaussian) error propagation, and scientific rounding.*
     - [GUM Tree Calculator](https://github.com/MSLNZ/GTC): *A Python package for processing data with measurement uncertainty.*
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlations and limits of the approximation

    For correlated inputs, the first-order variance is $u_f^2=\nabla f^T C\nabla f$, where $C$ is the input covariance matrix. The sum of squared terms earlier assumes zero covariance. Reusing the same uncertain object preserves its dependence; creating a new object describes a distinct input.

    Near a turning point the first derivative can vanish even though the output varies. That is a warning to inspect higher-order effects or a simulation of the stated input distributions. Standard uncertainty is not an absolute error bound and is not automatically a 95% interval.
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## In conclusion

    There is no excuse to not perform propagation of uncertainty of the result of a measurement. Remember, your work is complete only when you report the uncertainty.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Is uncertainty the same as a known measurement error?
    2. When can covariance terms be omitted?
    3. Does a one-SD uncertainty automatically mean 95% coverage?

    /// details | Checkpoint answers
    1. No; it quantifies dispersion consistent with available knowledge.
    2. When input covariances are zero or justifiably negligible.
    3. No.
    ///

    ## Problems

    1. Give the density and standard uncertainty for the original independent measurements.
    2. For independent lengths $a=3.0\pm0.1$ cm and $b=4.0\pm0.2$ cm, give the standard uncertainty of $a+b$.
    3. For the cylinder example, give volume and first-order standard uncertainty.

    /// details | Final answers
    1. $2.00\pm0.32$ g/cm³.
    2. Approximately 0.224 cm.
    3. Approximately $31.42\pm3.51$ cm³.
    ///

    ## Go deeper

    [JCGM GUM publications](https://www.bipm.org/en/committees/jc/jcgm/publications) — measurement models and uncertainty. [uncertainties technical guide](https://uncertainties.readthedocs.io/en/latest/tech_guide.html) — correlations and local linearization.

    ### Video lectures on the Internet

    [MIT Statistics for Applications lectures](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/video_galleries/lecture-videos/) — lectures on estimation provide a useful comparison with measurement uncertainty.
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Evaluation of measurement data - Guide to the expression of uncertainty in measurement](https://doi.org/10.59161/JCGM100-2008E). JCGM 100:2008.

    - [Avaliação de dados de medição - Guia para a expressão de incerteza de medição](http://www.inmetro.gov.br/inovacao/publicacoes/gum_final.pdf). GUM 2008 in Portuguese.

    - [uncertainties](https://uncertainties.readthedocs.io/en/latest/): a Python package for calculations with uncertainties, Eric O. Lebigot.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
