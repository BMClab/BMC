# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic trigonometry

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Imagine looking at a straight forearm from the side. You know its length
    and its angle above the horizontal. How far is the hand in front of the
    elbow, and how far above it? Trigonometry connects those measurements.

    We will start with the ratios of a right triangle, connect degrees to
    radians, and use a few identities to check our calculations. You need
    only ratios, square roots, and elementary algebra.

    Work through the six **Exercises** on a scratchpad before opening their
    solutions. Each solution is collapsed just below its question. When you
    reach the angle slider, predict what will change before moving it.

    At the end, use the **Checkpoint questions** to check your understanding,
    then try the **Problems** independently. Problems have collapsed final
    answers, without derivations. **Go deeper** and **Video lectures on the
    Internet** offer optional follow-up material; **References** lists the
    sources you can consult.

    You can use this notebook locally, in [marimo.app](https://marimo.app/),
    or in [molab](https://molab.marimo.io/notebooks). The examples and diagrams
    are contained in this file; there are no data files to download. In an
    editor, run the cells to activate the slider and numerical checks.

    **Challenge 0.** Sketch a forearm at $30^\circ$ above the horizontal,
    then at $60^\circ$, keeping its length fixed. Predict how the horizontal
    and vertical distances from elbow to hand change. Keep your sketch;
    we will return to it in the checkpoint questions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    Python's standard `math` module supplies the trigonometric functions.
    Marimo displays the text, diagrams, slider, and expandable answers.
    """)
    return


@app.cell
def _():
    import math

    import marimo as mo

    return math, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Same angle, same ratios

    Start with a right triangle: one angle is $90^\circ$ ($\pi/2$ radians).
    Choose one of its acute angles and call it $\theta$. The **hypotenuse**
    is opposite the right angle. Of the two remaining sides, one is
    **opposite** $\theta$ and the other is **adjacent** to it.

    If we enlarge the triangle without changing its angles, all three sides
    grow by the same factor. Their ratios stay the same. Those ratios define
    the trigonometric functions:

    $$
    \sin\theta = \frac{\text{opposite}}{\text{hypotenuse}},\qquad
    \cos\theta = \frac{\text{adjacent}}{\text{hypotenuse}},\qquad
    \tan\theta = \frac{\text{opposite}}{\text{adjacent}}.
    $$

    Keep track of the angle you chose: changing to the other acute angle
    exchanges the opposite and adjacent sides.

    **Guiding questions.** What happens to these ratios if every side is
    measured in millimetres instead of centimetres? Can the sine of an
    acute angle be larger than one? Use the triangle to explain your answers
    before continuing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <svg viewBox="0 0 460 270" width="100%" style="max-width:460px"
         role="img" aria-label="Right triangle with theta at the lower left,
         adjacent side along the bottom, opposite side at the right, and hypotenuse sloping upward.">
      <g fill="none" stroke="currentColor" stroke-width="2">
        <path d="M 55 215 L 345 215 L 345 45 Z"/>
        <path d="M 329 215 L 329 199 L 345 199"/>
        <path d="M 100 215 A 45 45 0 0 0 94 192"/>
      </g>
      <g fill="currentColor" font-family="sans-serif" font-size="16">
        <text x="110" y="202">θ</text>
        <text x="155" y="245">adjacent</text>
        <text x="355" y="135">opposite</text>
        <text x="150" y="105">hypotenuse</text>
      </g>
    </svg>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1 — Enlarge the triangle.** The opposite, adjacent, and
    hypotenuse lengths are 3, 4, and 5 cm. Find $\sin\theta$, $\cos\theta$,
    and $\tan\theta$. Now double all three lengths. Which ratios change?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Solution to Exercise 1": mo.md(r"""
    The ratios are $\sin\theta=3/5=0.6$, $\cos\theta=4/5=0.8$,
    and $\tan\theta=3/4=0.75$. None has a unit: the length units cancel.

    After doubling the sides, the ratios are $6/10$, $8/10$, and $6/8$,
    respectively. All are unchanged. The angle determines the ratios;
    the size of the triangle does not.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### From a length to two distances

    Return to the forearm. Represent it by a straight segment of length $L$
    at an acute angle $\theta$ above the horizontal. Its horizontal distance
    $x$ is the adjacent side, and its vertical distance $y$ is the opposite
    side. Multiplying the ratios by $L$ gives

    $$x=L\cos\theta,\qquad y=L\sin\theta.$$

    **Exercise 2 — Locate the hand.** For $L=0.30$ m and
    $\theta=30^\circ$, find $x$ and $y$. Use
    $\sin30^\circ=1/2$ and $\cos30^\circ=\sqrt{3}/2$.
    Which distance is larger, and does that fit your sketch?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Solution to Exercise 2": mo.md(r"""
    $$x=0.30\frac{\sqrt{3}}{2}\approx0.260\ \text{m},\qquad
    y=0.30\frac{1}{2}=0.150\ \text{m}.$$

    The horizontal distance is larger. At $30^\circ$ above the
    horizontal the segment is closer to horizontal than vertical,
    which agrees with the sketch. These are distances from the elbow,
    not absolute coordinates in the laboratory.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Radians: measure an angle with an arc

    Why use radians when degrees already describe angles? Radians connect
    an angle directly to lengths. Draw a circle of radius $r$ and an arc
    of length $s$. The angle subtended by that arc, in radians, is

    $$\theta=\frac{s}{r}.$$

    An arc as long as the radius gives an angle of **1 radian**. A full
    circle has arc length $2\pi r$, so a full turn is $2\pi$ radians or
    $360^\circ$. Consequently,

    $$\theta_{\mathrm{rad}}=\theta_{\mathrm{deg}}\frac{\pi}{180},\qquad
    \theta_{\mathrm{deg}}=\theta_{\mathrm{rad}}\frac{180}{\pi}.$$

    The ratio $s/r$ is dimensionless, but writing “rad” helps distinguish
    it from a number expressed in degrees.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <svg viewBox="0 0 420 260" width="100%" style="max-width:420px"
         role="img" aria-label="Circular sector of one radian: both radii
         have length r and the connecting circular arc has length s equal to r.">
      <g fill="none" stroke="currentColor" stroke-width="2">
        <path d="M 65 215 L 245 215 M 65 215 L 162.25 63.54"/>
        <path d="M 245 215 A 180 180 0 0 0 162.25 63.54" stroke-width="4"/>
        <path d="M 105 215 A 40 40 0 0 0 86.61 181.34"/>
      </g>
      <g fill="currentColor" font-family="sans-serif" font-size="16">
        <text x="150" y="240">r</text>
        <text x="90" y="130">r</text>
        <text x="253" y="130">arc length s = r</text>
        <text x="115" y="185">1 rad</text>
      </g>
    </svg>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3 — Arc length and angle.** An arc is 0.20 m long on a
    circle of radius 0.40 m. What angle does it subtend, in radians and
    degrees? Separately, convert $60^\circ$ to radians. If you double both
    the radius and arc length, does the angle change?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Solution to Exercise 3": mo.md(r"""
    The first angle is $0.20/0.40=0.50$ rad, or
    $0.50\times180/\pi\approx28.65^\circ$.
    The second is $60\pi/180=\pi/3\approx1.0472$ rad.

    Doubling both lengths leaves their ratio unchanged:
    $0.40/0.80=0.50$ rad. Just as with similar triangles, scaling
    the drawing does not change its angle.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Common trigonometric values

    Here are useful exact values. At $0^\circ$ and $90^\circ$, read the
    sine and cosine entries as the limiting values of the triangle ratios.

    | $\theta$ (degrees) | $\theta$ (radians) | $\sin\theta$ | $\cos\theta$ | $\tan\theta$ |
    | --- | --- | --- | --- | --- |
    | $0^\circ$ | $0$ | $0$ | $1$ | $0$ |
    | $30^\circ$ | $\pi/6$ | $1/2$ | $\sqrt{3}/2$ | $1/\sqrt{3}$ |
    | $45^\circ$ | $\pi/4$ | $\sqrt{2}/2$ | $\sqrt{2}/2$ | $1$ |
    | $60^\circ$ | $\pi/3$ | $\sqrt{3}/2$ | $1/2$ | $\sqrt{3}$ |
    | $90^\circ$ | $\pi/2$ | $1$ | $0$ | undefined |

    Notice that $\tan\theta=\sin\theta/\cos\theta$ wherever the
    denominator is nonzero. At exactly $90^\circ$, tangent is undefined:
    division by zero does not give a real number. As an acute angle
    approaches $90^\circ$, its tangent grows without bound.

    ### Check a prediction with Python

    Python's mathematical functions expect **radians**. We convert the
    slider's angle before evaluating them. Before moving the slider from
    $30^\circ$ to $60^\circ$, predict which two values exchange places.
    Check your prediction against the exact-value table.
    """)
    return


@app.cell
def _(mo):
    angle_degrees = mo.ui.slider(
        start=0, stop=90, step=1, value=30,
        label="Angle (degrees)", show_value=True,
    )
    angle_degrees
    return (angle_degrees,)


@app.cell
def _(angle_degrees, math, mo):
    _degrees = angle_degrees.value
    _radians = math.radians(_degrees)
    _sine = math.sin(_radians)
    _cosine = math.cos(_radians)
    # Handle the exact endpoint explicitly instead of dividing by roundoff.
    _tangent = "undefined" if _degrees == 90 else f"{math.tan(_radians):.6f}"
    mo.md(f"""
    At **{_degrees}°** ({_radians:.6f} rad):

    | Sine | Cosine | Tangent |
    | --- | --- | --- |
    | {_sine:.6f} | {_cosine:.6f} | {_tangent} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4 — Check the table, not just the calculator.** Use the
    sine and cosine entries to calculate $\tan30^\circ$ exactly. Then
    explain why a computer might return a very large finite number when
    computing the tangent of its stored approximation of $\pi/2$, even
    though $\tan90^\circ$ is undefined.
    Hint: a computer stores only a finite-precision approximation of $\pi$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Solution to Exercise 4": mo.md(r"""
    $$\tan30^\circ=\frac{1/2}{\sqrt{3}/2}
    =\frac{1}{\sqrt{3}}\approx0.57735.$$

    The stored approximation of $\pi/2$ is not the exact angle.
    Its computed cosine is close to zero, rather than exactly zero,
    and the computed tangent can be enormous. That finite result is
    not a mathematical value for $\tan90^\circ$. The slider handles
    its known $90^\circ$ endpoint explicitly and displays “undefined”.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Trigonometric identities: another way to check

    An identity is an equality that holds for every angle in its domain.
    These identities let us calculate a value in more than one way:

    $$\sin^2\alpha+\cos^2\alpha=1,$$

    $$\sin(2\alpha)=2\sin\alpha\cos\alpha,$$

    $$\cos(2\alpha)=\cos^2\alpha-\sin^2\alpha,$$

    $$\sin(\alpha\pm\beta)
      =\sin\alpha\cos\beta\pm\cos\alpha\sin\beta,$$

    $$\cos(\alpha\pm\beta)
      =\cos\alpha\cos\beta\mp\sin\alpha\sin\beta.$$

    In the last identity, adding the angles means subtracting the sine
    product; subtracting the angles means adding it. Both factors in that
    product are sines. Setting $\beta=\alpha$ recovers the double-angle
    identities above.

    **Exercise 5 — Build an unfamiliar value from familiar ones.** Use
    $75^\circ=45^\circ+30^\circ$ and the table to find exact expressions
    for $\sin75^\circ$ and $\cos75^\circ$. Then use the cosine
    double-angle identity at $\alpha=45^\circ$ to recover $\cos90^\circ$.
    """)
    return


@app.cell(hide_code=True)
def _(math, mo):
    _angle = math.radians(75)
    mo.accordion(
        {"Solution to Exercise 5": mo.vstack([mo.md(r"""
    $$\sin75^\circ
      =\frac{\sqrt{2}}{2}\frac{\sqrt{3}}{2}
       +\frac{\sqrt{2}}{2}\frac{1}{2}
      =\frac{\sqrt{6}+\sqrt{2}}{4}\approx0.965926.$$

    $$\cos75^\circ
      =\frac{\sqrt{2}}{2}\frac{\sqrt{3}}{2}
       -\frac{\sqrt{2}}{2}\frac{1}{2}
      =\frac{\sqrt{6}-\sqrt{2}}{4}\approx0.258819.$$

    For the double angle,
    $\cos90^\circ=(\sqrt{2}/2)^2-(\sqrt{2}/2)^2=0$.
    Notice that the sine at $75^\circ$ is larger than the cosine,
    as expected for a segment closer to vertical than horizontal.
    """), mo.md(f"""
        **Numerical check with Python.** The two ways of computing each
        value agree to the six decimal places displayed:

        | Value | Trigonometric function | Expression with square roots |
        | --- | --- | --- |
        | Sine of 75° | {math.sin(_angle):.6f} | {(math.sqrt(6) + math.sqrt(2)) / 4:.6f} |
        | Cosine of 75° | {math.cos(_angle):.6f} | {(math.sqrt(6) - math.sqrt(2)) / 4:.6f} |
        """)])},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Put the pieces together

    **Exercise 6 — Check a reconstructed length.** A straight segment is
    0.50 m long and makes an angle of $60^\circ$ above the horizontal.
    Find its horizontal and vertical distances, $x$ and $y$, then check
    that $x^2+y^2=L^2$. Which identity explains why this check works for
    every acute angle? If the angle changes to $30^\circ$ while the
    length stays fixed, predict the new distances without starting again.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Solution to Exercise 6": mo.md(r"""
    At $60^\circ$, $x=0.50/2=0.25$ m and
    $y=0.50\sqrt{3}/2=\sqrt{3}/4\approx0.433$ m.

    $$x^2+y^2=\frac{1}{16}+\frac{3}{16}
    =\frac{1}{4}=L^2\quad\text{(in m²)}.$$

    More generally,
    $x^2+y^2=L^2(\cos^2\theta+\sin^2\theta)=L^2$.
    Use the unrounded values when checking this numerically.

    At $30^\circ$, sine and cosine exchange values: the horizontal
    distance becomes approximately 0.433 m and the vertical distance
    becomes 0.25 m. The segment's length is still 0.50 m.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems. Try to answer without scrolling back.

    1. Return to your sketch from Challenge 0. As the forearm moves from
       $30^\circ$ to $60^\circ$, which distance increases and which decreases?
       What happens to the length of the forearm?
    2. Why do similar triangles give the same trigonometric ratios even
       when their side lengths differ? Does changing the length unit alter
       those ratios? Can the sine of an acute angle exceed one?
    3. What does an angle of one radian mean geometrically? What must you do
       before passing an angle measured in degrees to Python's `math.sin`?
    4. Why is $\tan90^\circ$ undefined, even if a numerical calculation
       returns a very large value?
    5. In the cosine angle-sum identity, which sign connects the two products,
       and which functions appear in the second product?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Answers to the checkpoint questions": mo.md(r"""
    1. The vertical distance increases, the horizontal distance decreases,
       and the two distances exchange values. The forearm's length stays
       the same.
    2. All sides scale by the same factor, which cancels in each ratio.
       Changing the length unit also leaves the ratios unchanged. For an
       acute angle, the opposite side is shorter than the hypotenuse,
       so its sine is less than one.
    3. The arc length equals the radius. Convert degrees to radians,
       for example with `math.radians`, before evaluating `math.sin`.
    4. The exact cosine is zero, so the ratio is undefined. A computer
       uses an approximation of the angle and need not obtain an exact
       zero for its cosine.
    5. Subtraction: the second product is $\sin\alpha\sin\beta$.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    Use these for independent practice. Sketch the geometry, keep track of
    units, and check your result before opening an answer. Only final
    answers are provided here; the earlier exercises contain worked solutions.

    **Problem 1 — Work backwards from distances.** A straight segment has
    horizontal and vertical distances of 0.24 m and 0.18 m, respectively.
    Find its length and the sine, cosine, and tangent of its acute angle
    above the horizontal. You do not need to calculate the angle itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Answer to Problem 1": mo.md(r"""
    $L=0.30$ m; $\sin\theta=0.60$;
    $\cos\theta=0.80$; $\tan\theta=0.75$.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2 — Use an angle difference.** Use $15^\circ=45^\circ-30^\circ$
    to find exact expressions for $\sin15^\circ$ and $\cos15^\circ$.
    Check their approximate values with the slider.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Answer to Problem 2": mo.md(r"""
    $\sin15^\circ=(\sqrt{6}-\sqrt{2})/4\approx0.258819$.

    $\cos15^\circ=(\sqrt{6}+\sqrt{2})/4\approx0.965926$.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 3 — A length check that misses a mistake.** A colleague wants
    the horizontal and vertical distances of a 0.40 m segment at $60^\circ$
    above the horizontal. They calculate `x = 0.40 * math.cos(60)` and
    `y = 0.40 * math.sin(60)`. Identify the mistake and give the correct
    distances. Would checking only that $x^2+y^2=0.40^2$ reveal the mistake?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Answer to Problem 3": mo.md(r"""
    The code interprets 60 as radians; use `math.radians(60)`.
    The correct distances are $x=0.200$ m and
    $y=\sqrt{3}/5\ \text{m}\approx0.346410$ m.
    The length check alone does **not** reveal the mistake.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Choose a resource that answers a question you still have. These are
    optional next steps, not prerequisites for the problems above.

    - [Proofs of trigonometric identities](https://en.wikipedia.org/wiki/Proofs_of_trigonometric_identities): revisit the angle-sum and double-angle identities and see how they can be derived.
    - [Scalar and vector](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/scalar_vector.py): connect the horizontal and vertical distances in this guide to vector components.
    - [Rigid-body transformations in 2D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_2d.py): see where the same sines and cosines appear when coordinates are rotated.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Video lectures on the Internet

    - [Basic trigonometry — Khan Academy](https://www.youtube.com/watch?v=F21S9Wpi0y8): a visual introduction to sine, cosine, and tangent in right triangles.
    - [Introduction to radians — Khan Academy](https://www.khanacademy.org/math/geometry/hs-geo-trig/hs-geo-trig-functions/v/introduction-to-radians): connect arc length, radians, and degrees.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Trigonometric functions — Wikipedia](https://en.wikipedia.org/wiki/Trigonometric_functions).

    - [Radian — Wikipedia](https://en.wikipedia.org/wiki/Radian).

    - [Trigonometric identities — Wikipedia](https://en.wikipedia.org/wiki/List_of_trigonometric_identities).

    - [Trigonometry — Coddy](https://coddy.tech/landing/trigonometry).

    - [Python's mathematical functions](https://docs.python.org/3/library/math.html#trigonometric-functions).
    """)
    return


if __name__ == "__main__":
    app.run()
