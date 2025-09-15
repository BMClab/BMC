import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Propagation of uncertainty (error)

        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Evaluation-of-measurement-data" data-toc-modified-id="Evaluation-of-measurement-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Evaluation of measurement data</a></span></li><li><span><a href="#Propagation-of-uncertainty-by-linear-approximation" data-toc-modified-id="Propagation-of-uncertainty-by-linear-approximation-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Propagation of uncertainty by linear approximation</a></span></li><li><span><a href="#Using-the-uncertainties-package-to-calculate-uncertainty" data-toc-modified-id="Using-the-uncertainties-package-to-calculate-uncertainty-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Using the <code>uncertainties</code> package to calculate uncertainty</a></span><ul class="toc-item"><li><span><a href="#Displaying-the-output-in-different-formats" data-toc-modified-id="Displaying-the-output-in-different-formats-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Displaying the output in different formats</a></span></li><li><span><a href="#Automatic-deduction-of-the-symbolic-formula-for-error-propagation" data-toc-modified-id="Automatic-deduction-of-the-symbolic-formula-for-error-propagation-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Automatic deduction of the symbolic formula for error propagation</a></span></li><li><span><a href="#Another-example" data-toc-modified-id="Another-example-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Another example</a></span></li></ul></li><li><span><a href="#Other-Python-packages-for-uncertainty-analysis" data-toc-modified-id="Other-Python-packages-for-uncertainty-analysis-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Other Python packages for uncertainty analysis</a></span></li><li><span><a href="#In-conclusion" data-toc-modified-id="In-conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>In conclusion</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Evaluation of measurement data

        > The result of a measurement is only an approximation or estimate of the value of the measurand and thus is complete only when accompanied by a statement of the uncertainty of that estimate.   
        >   
        > Uncertainty (of measurement) is a parameter, associated with the result of a measurement, that characterizes the dispersion of the values that could reasonably be attributed to the measurand.   
        >   
        > [Evaluation of measurement data - Guide to the expression of uncertainty in measurement (2008)](https://www.bipm.org/utils/common/documents/jcgm/JCGM_100_2008_E.pdf)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For instance, to calculate the density of a material someone made the following measurements of mass and volume (m=d/V):$m = 4.0 \pm 0.5 \; g$$V = 2.0 \pm 0.2 \; cm^3$Where 0.5 g and 0.2 cm$^3$each represent a value of one standard deviation and express the errors in the measured weight and volume.   

        Based on these numbers, the material density is 2 g/cm$^3$; but how much is the error (uncertainty) of the density?   
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One way to estimate the error in the density is by propagation of uncertainty:

        > Propagation of uncertainty (or propagation of error) is the effect of variables' uncertainties (or errors) on the uncertainty of a function based on them. When the variables are the values of experimental measurements they have uncertainties due to measurement limitations (e.g., instrument precision and noise) which propagate to the combination of variables in the function ([Wikipedia](http://en.wikipedia.org/wiki/Propagation_of_uncertainty)).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Propagation of uncertainty by linear approximation

        The classic way for propagation of uncertainty is through linear approximation of error propagation theory. We start with the mathematical function that relates the physical quantities and estimate the contribution of their variations by means of partial derivatives (the rate of variation) of each of these primary quantities. The linearity of the mathematical function in the vicinity of the obtained value and the statistical correlations between these quantities are also taken into consideration in the propagation of uncertainty.   

        If the measurand$f$is defined in terms of the variables$x, y, z, ...$by a general expression$f(x, y, z, ...)$, a first order approximation of the propagation of uncertainty ignoring correlations between these variables is given by the well known formula:$\sigma _{f}\;=\;\sqrt{\left ( \frac{\partial f}{\partial x}  \right)^2\sigma _{x}^{2} + \left ( \frac{\partial f}{\partial y} \right)^2 \sigma _{y}^{2} + \left ( \frac{\partial f}{\partial z} \right)^2 \sigma _{z}^{2}\: + \ldots}$Where$\partial f/\partial i$is the first partial derivative of$f$with respect to variable$i$and$\sigma_i$is the uncertainty of the measurement of the variable$i$(one standard deviation).  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Using the former formula for linear approximation of error propagation, the estimation of the uncertainty of the density is given by:$d=\frac{m}{V} \\
        \sigma _{d}\;=\;\sqrt{ \left ( \frac{\partial d}{\partial m}  \right)^2 \sigma _{m}^{2}+\left ( \frac{\partial d}{\partial V} \right)^2 \sigma _{V}^{2} } \\
        \sigma _{d}\;=\;\sqrt{ \left ( \frac{1}{V} \right)^2 \sigma _{m}^{2}+\left ( -\frac{m}{V^2} \right)^2 \sigma _{V}^{2} }$And considering the values of mass and volume given earlier:$\sigma_{d}\;=\;\sqrt{\left(\frac{1}{2}\right)^2\times 0.5\:^2+\left(-\frac{4}{2^2}\right)^2\times 0.2\:^2}\;\;\approx\; 0.32$Then, the estimated value of the density can now be expressed in a complete form: d = 2.00±0.32 g/cm$^3$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Using the `uncertainties` package to calculate uncertainty

        It is essential that any result of a measurement is accompanied by the estimate of its uncertainty as above, this way we can better understand the value of the measure and how reliable it is. However, it is unlikely that anyone will repeat these calculations in the day-to-day work (although there are tables with partial derivatives for the most common expressions). Therefore, it is useful a software that can perform these calculations (at least for checking the results manually obtained). Fortunately, there are software for the propagation of uncertainty, [see this list on Wikipedia](https://en.wikipedia.org/wiki/List_of_uncertainty_propagation_software).   

        One such software, free, open source, and cross-platform, is the library for [Python](https://python.org) called [uncertainties](https://pypi.org/project/uncertainties/). With [uncertainties](https://pypi.org/project/uncertainties/) installed, let's calculate again the uncertainty of the density:
        """
    )
    return


@app.cell
def _():
    # Import the necessary libraries
    from IPython.display import display, Math  # IPython formatting
    from uncertainties import ufloat           # to define variables with uncertainty
    return Math, display, ufloat


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's input the values:
        """
    )
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
    mo.md(
        r"""
        The result we obtained before, but without having to deduct the partial derivatives because [uncertainties](http://pythonhosted.org/uncertainties/) does this with an [algorithm for automatic differentiation](http://en.wikipedia.org/wiki/Automatic_differentiation).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Displaying the output in different formats
        """
    )
    return


@app.cell
def _(Math, d, display):
    display(Math(f'd = {d:.3L} g/cm^3'))
    display(Math(f'd = {d:.2eL} g/cm^3'))
    display(Math(f'd = {d:.2uS} g/cm^3'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The operation above could have been performed in a single line (after you imported the necessary libraries at the first time using them):
        """
    )
    return


@app.cell
def _(ufloat):
    print(ufloat(4.0, 0.5) / ufloat(2.0, 0.2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And how much each variable mass and volume contributed to the final uncertainty (i.e., the partial derivative of density in relation to the variable mass or volume times its standard deviation (each term on the right side of the equation above without squaring)) is given by:
        """
    )
    return


@app.cell
def _(d):
    d.error_components().items()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        By definition, according to the formula above, the square root of the quadratic sum of these values is the total uncertainty for the density.   

        The value of each partial derivative of the density function with respect to the mass and volume variables is given by:
        """
    )
    return


@app.cell
def _(d):
    d.derivatives
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Automatic deduction of the symbolic formula for error propagation 

        Stepping back, we can even deduce the formula for the error propagation using the partial derivatives in case we want the actual formula using [Sympy](http://sympy.org/en/index.html):
        """
    )
    return


@app.cell
def _(Math, display):
    from sympy import symbols, Symbol, sqrt, Add, Mul, Pow, init_printing, latex
    init_printing()

    def stdform(formula, formula_symb, *errors):
        """
        Calculate the symbolic formula for error propagation using partial derivatives.
        """
        from functools import reduce
        formula2 = sqrt(reduce(Add, (Mul(Pow(_formula.diff(var), 2, evaluate=False), Symbol(f'\\sigma_{var.name}', commutative=False) ** 2, evaluate=False) for var in _formula.atoms(Symbol) if var.name in errors), 0.0))
        formula3 = sqrt(reduce(Add, (Mul(Pow(_formula.diff(var) / _formula, 2, evaluate=False), Symbol(f'\\sigma_{var.name}', commutative=False) ** 2, evaluate=False) for var in _formula.atoms(Symbol) if var.name in errors), 0.0))
        print('Uncertainty:')
        display(Math(f'\\sigma_{formula_symb} = {latex(formula2)}'))
        print('Relative Uncertainty:')
        display(Math(f'\\frac{{\\sigma_{formula_symb}}}{formula_symb} = {latex(formula3)}'))
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
    mo.md(
        r"""
        Same as before, but Sympy deduced for us.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Another example

        Let's estimate the uncertainty in the measurement of the volume of a cylinder given its diameter ($d=2.0\pm0.1$) cm and height$h=10.0\pm0.5$cm. But now, let's go straight to the solution:
        """
    )
    return


@app.cell
def _(stdform, symbols):
    from sympy import pi
    (d_2, _h) = symbols('d, h')
    _V = _h * _pi * (d_2 / 2) ** 2
    _formula = stdform(_V, 'V', 'd', 'h')
    return


@app.cell
def _(Math, display, ufloat):
    from math import pi
    _h = ufloat(nominal_value=10.0, std_dev=0.5)
    d_3 = ufloat(nominal_value=2.0, std_dev=0.1)
    _V = _h * _pi * (d_3 / 2) ** 2
    display(Math(f'V = {_V:.3L} cm^3'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Other Python packages for uncertainty analysis

        The uncertainty package employs a linear approximation of error propagation, which is only valid for small errors, and it assumes that the errors follow normal distributions. There are other packages that can be used when these assumptions are violated:  
         - [NIST Uncertainty Machine](https://uncertainty.nist.gov/): *A Web-based software application to evaluate the measurement uncertainty associated with an output quantity defined by a measurement model of the form$y=f(x_0,	\ldots,x_n)$.*  The NIST Uncertainty Machine allows to select a probability distribution for each of the input quantities and can evaluate the measurement uncertainty using two methods: the linear approximation and the Monte Carlo method,
         - [mcerp](https://github.com/tisimst/mcerp): *A stochastic calculator for Monte Carlo methods that uses latin-hypercube sampling to perform non-order specific error propagation (or uncertainty analysis).*  
         - [soerp](https://github.com/tisimst/soerp): *Python implementation of the original Fortran code SOERP by N. D. Cox to apply a second-order analysis to error propagation (or uncertainty analysis). The soerp package allows you to easily and transparently track the effects of uncertainty through mathematical calculations.*   
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And there are other packages using linear error propagation:  

         - [scinum](https://github.com/riga/scinum): *Provides a simple Number class that wraps plain floats or NumPy arrays and adds support for multiple uncertainties, automatic (gaussian) error propagation, and scientific rounding.*  
         - [GUM Tree Calculator](https://github.com/MSLNZ/GTC): *A Python package for processing data with measurement uncertainty.*  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## In conclusion

        There is no excuse to not perform propagation of uncertainty of the result of a measurement. Remember, your work is complete only when you report the uncertainty.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - [Evaluation of measurement data - Guide to the expression of uncertainty in measurement](https://www.bipm.org/utils/common/documents/jcgm/JCGM_100_2008_E.pdf). JCGM 100:2008.  
        - [Avaliação de dados de medição - Guia para a expressão de incerteza de medição](http://www.inmetro.gov.br/inovacao/publicacoes/gum_final.pdf). GUM 2008 in Portuguese.
        - [uncertainties](https://pythonhosted.org/uncertainties/): a Python package for calculations with uncertainties, Eric O. Lebigot.   
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
