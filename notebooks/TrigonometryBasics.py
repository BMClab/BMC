import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Basic trigonometry

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
        If two right triangles (triangles with an angle of$90^o$($\pi/2$radians)) have equal acute angles, they are similar, so their side lengths are proportional.  
        These proportionality constants are the values of$\sin\theta$,$\cos\theta$, and$\tan\theta$.  
        Here is a geometric representation of the main [trigonometric functions](http://en.wikipedia.org/wiki/Trigonometric_function) of an angle$\theta$:  
        <br>
        <figure><img src='http://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Academ_Base_of_trigonometry.svg/300px-Academ_Base_of_trigonometry.svg.png' alt='Main trigonometric functions'/><figcaption><center><i>Figure. Main trigonometric functions (<a href="http://en.wikipedia.org/wiki/Trigonometric_function">image from Wikipedia</a>).</i></center></figcaption></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Radian

        An arc of a circle with the same length as the radius of that circle corresponds to an angle of 1 radian:  
        <br>
        <figure><img src='http://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Radian_cropped_color.svg/220px-Radian_cropped_color.svg.png' width=200/><figcaption><center><i>Figure. Definition of the radian (<a href="https://en.wikipedia.org/wiki/Radian">image from Wikipedia</a>).</i></center></figcaption></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Common trigonometric values

        <table>
        <tr>
        <th style="text-align: center; background-color:#FBFBEF">$\;\theta\;(^o)$</th>
        <th style="text-align: center; background-color:#FBFBEF">$\;\theta\;(rad)$</th>
        <th style="text-align: center; background-color:#FBFBEF">$\;\sin \theta\;$</th>
        <th style="text-align: center; background-color:#FBFBEF">$\;\cos \theta\;$</th>
        <th style="text-align: center; background-color:#FBFBEF">$\;\tan \theta\;$</th>
        </tr>
        <tr>
        <td style="text-align: center">$0^o$</td>
        <td style="text-align: center">$0$</td>
        <td style="text-align: center">$0$</td>
        <td style="text-align: center">$1$</td>
        <td style="text-align: center">$0$</td>
        </tr>
        <tr>
        <td style="text-align: center">$30^o$</td>
        <td style="text-align: center">$\pi/6$</td>
        <td style="text-align: center">$1/2$</td>
        <td style="text-align: center">$\sqrt{3}/2$</td>
        <td style="text-align: center">$1\sqrt{3}$</td>
        </tr>
        <tr>
        <td style="text-align: center">$45^o$</td>
        <td style="text-align: center">$\pi/4$</td>
        <td style="text-align: center">$\sqrt{2}/2$</td>
        <td style="text-align: center">$\sqrt{2}/2$</td>
        <td style="text-align: center">$1$</td>
        </tr>
        <tr>
        <td style="text-align: center">$60^o$</td>
        <td style="text-align: center">$\pi/3$</td>
        <td style="text-align: center">$\sqrt{3}/2$</td>
        <td style="text-align: center">$1/2$</td>
        <td style="text-align: center">$\sqrt{3}$</td>
        </tr>
        <tr>
        <td style="text-align: center">$90^o$</td>
        <td style="text-align: center">$\pi/2$</td>
        <td style="text-align: center">$1$</td>
        <td style="text-align: center">$0$</td>
        <td style="text-align: center">$\infty$</td>
        </tr>
        </table>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Trigonometric identities

        Some of the main trigonometric identities are (see a [complete list at Wikipedia](http://en.wikipedia.org/wiki/List_of_trigonometric_identities)):$\sin^2{\alpha} + \cos^2{\alpha} = 1$$\sin(2\alpha) = 2\sin{\alpha} \cos{\alpha}$$\cos(2\alpha) = \cos^2{\alpha} - \sin^2{\alpha}$$\sin(\alpha \pm \beta) = \sin{\alpha} \cos{\beta} \pm \cos{\alpha} \sin{\beta}$$\cos(\alpha \pm \beta) = \cos{\alpha} \cos{\beta} \mp \sin{\alpha} \cos{\beta}$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - [Trigonometric functions [Wikipedia]](http://en.wikipedia.org/wiki/Trigonometric_function).
        - [Trigonometry [S.O.S. Mathematics]](http://www.sosmath.com/trig/trig.html).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
