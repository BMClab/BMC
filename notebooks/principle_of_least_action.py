import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Principle of least action

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
        The [principle of least action](https://en.wikipedia.org/wiki/Principle_of_least_action) applied to the movement of a mechanical system states that "the average kinetic energy less the average potential energy is as little as possible for the path of an object going from one point to another" (Prof. Richard Feynman in [The Feynman Lectures on Physics](http://www.feynmanlectures.caltech.edu/II_19.html)). 

        This principle is so fundamental that it can be used to derive the equations of motion of a system, for example, independent of the Newton's laws of motion. Let's now see the principle of least action in mathematical terms.

        The difference between the kinetic and potential energy in a system is known as the [Lagrange or Lagrangian function](https://en.wikipedia.org/wiki/Lagrangian_mechanics):$\mathcal{L} = T - V$The [principle of least action](https://en.wikipedia.org/wiki/Principle_of_least_action) states that the actual path which a system follows between two points in the time interval$t_1$and$t_2$is such that the integral$\mathcal{S}\; =\; \int _{t_1}^{t_2} \mathcal{L} \; dt$is stationary, meaning that$\delta \mathcal{S}=0$(i.e., the value of$\mathcal{S}$is an extremum), and it can be shown in fact that the value of this integral is a minimum for the actual path of the system. The integral above is known as the action integral and$\mathcal{S}$is known as the action.

        A formal demonstration that the the value of this integral above is stationary and that one can derive the Euler-Lagrange equation in the present form we use today was first formulated by the mathematician [William Hamilton](https://en.wikipedia.org/wiki/William_Rowan_Hamilton) in the XIXth century; because that the principle of least action as presented above is also known as [Hamilton's principle](https://en.wikipedia.org/wiki/Hamilton%27s_principle).

        For a simple and didactic example where the integral above is stationary, see [The Feynman Lectures on Physics](http://www.feynmanlectures.caltech.edu/II_19.html).  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Feynman R, Leighton R, Sands M (2013) [The Feynman Lectures on Physics - HTML edition](http://www.feynmanlectures.caltech.edu/).  
        - Taylor JR (2005) [Classical Mechanics](https://books.google.com.br/books?id=P1kCtNr-pJsC). University Science Books.  

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
