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
    # Tutorial on Python for scientific computing

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil

    An interactive [marimo](https://marimo.io/) guide.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    This is a first encounter with Python inside Marimo. We will calculate a speed, name quantities, collect repeated measurements, and write a small reusable calculation. You need arithmetic, not prior programming. The scientific topics later in this collection introduce their own numerical libraries when needed.

    Read the examples in order, predict each result, and then run the cell. Try the distributed Exercises before opening their worked solutions. Finish with the checkpoint questions and independent Problems; their panels contain final answers only.

    **Challenge 0.** A walker covers 6 metres in 4 seconds. Predict the speed, then decide which value should change if the time doubles.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    Only Marimo and Python’s standard library are needed. Open this file in the editor, locally or on a Marimo service. Run a cell after editing it. Marimo reruns cells that depend on its variables, so keep each public variable defined in one cell. A name beginning with `_` is local to its cell.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Expressions and names

    Python can be a calculator: `+`, `-`, `*`, `/`, and `**` mean addition, subtraction, multiplication, division, and exponentiation. Parentheses control order. `=` assigns a name; it does not assert a mathematical equality. Units belong in names and in your explanation because ordinary numbers do not remember them.

    The first cell defines the inputs. The second depends on them. Change `time_s`, rerun its cell, and watch the downstream speed update.
    """)
    return


@app.cell
def _():
    distance_m = 6.0
    time_s = 4.0
    return distance_m, time_s

@app.cell
def _(distance_m, time_s):
    speed_m_s = distance_m / time_s
    print(f"Speed: {speed_m_s:.2f} m/s")
    return (speed_m_s,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** Predict `2 + 3 * 4`, `(2 + 3) * 4`, and `3 ** 2`. Then change the walking time from 4 to 8 s.

    /// details | Solution to Exercise 1
    Multiplication comes before addition, giving 14; parentheses give 20; exponentiation gives 9. Speed changes from $6/4=1.5$ to $6/8=0.75$ m/s. The distance input stays unchanged.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Values, types, and collections

    An integer such as `3`, a floating-point number such as `3.0`, a string such as `"trial 1"`, and a Boolean such as `True` are different types. `type(value)` tells you which one you have. A list stores a sequence; indexing starts at zero. A dictionary associates a key with a value.

    Read `trial_times_s[0]` as “the first trial time”, not “trial zero”. A slice such as `[1:3]` includes position 1 and stops before position 3. A tuple is an immutable sequence; a set stores distinct values. We will use lists for these repeated measurements.
    """)
    return


@app.cell
def _():
    trial_times_s = [4.0, 4.2, 3.8]
    participant = {"label": "P01", "distance_m": 6.0}
    print(trial_times_s[0], trial_times_s[1:3])
    print(participant["label"], len(trial_times_s))
    return participant, trial_times_s

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** For `[4.0, 4.2, 3.8]`, what do indices `0` and `-1` select? Why would `"4.0" + "4.2"` not calculate a total time?

    /// details | Solution to Exercise 2
    Index 0 selects 4.0 and index −1 selects the last item, 3.8. Quoted values are strings, so `+` concatenates them into `"4.04.2"`. Numeric addition requires numbers; `float("4.0")` converts a numeric string.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decisions and repetition

    A condition uses comparisons such as `>`, `<=`, or `==`. Combine conditions with `and`, `or`, and `not`. The indented lines after `if` belong to that branch. A `for` loop visits each item in a collection. `range(3)` supplies 0, 1, and 2; a `while` loop instead repeats while a condition remains true.

    Before running the next cell, count how many trials will pass the check. Nonpositive durations cannot define a physical walking speed.
    """)
    return


@app.cell
def _(participant, trial_times_s):
    for _duration in trial_times_s:
        if _duration > 0:
            print(f"{participant['distance_m'] / _duration:.3f} m/s")
        else:
            print("Duration must be positive")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** A loop visits `[4.0, 0.0, 5.0]`. How many speeds should it print using the condition above? What does `==` mean?

    /// details | Solution to Exercise 3
    Two: $6/4=1.5$ and $6/5=1.2$ m/s. The zero-duration trial follows the other branch. `==` compares values and produces a Boolean; `=` assigns a name.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Functions, imports, and errors

    A function gives a calculation a name, receives inputs, and returns a result. Keep units and valid inputs in its documentation. An exception reports a failed calculation; read the final error line and the line of code it points to. Do not replace an invalid measurement with a plausible-looking number.

    `import` is a statement that makes another module available. The standard `math` module includes square roots and trigonometric functions. `help(math.sqrt)` or the official documentation explains an unfamiliar function. Methods, such as a string’s `.lower()`, are operations attached to an object; you do not need to define classes to use this guide.
    """)
    return


@app.cell
def _():
    import math

    def walking_speed(distance_m, duration_s):
        """Return speed in m/s for a nonnegative distance and positive duration."""
        if distance_m < 0 or duration_s <= 0:
            raise ValueError("Use nonnegative distance and positive duration")
        return distance_m / duration_s

    print(walking_speed(10, 8))
    print(math.sqrt(9))
    return (walking_speed,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.** What should `walking_speed(10, 0)` do? Why return the computed value instead of only printing it?

    /// details | Solution to Exercise 4
    It should raise `ValueError`, because the duration is invalid. A returned number can be used in another calculation, stored, or plotted; `print` only displays a representation. Try valid inputs first, then inspect the exception from the invalid call in a separate scratch cell.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A control changes an input

    A slider is another way to supply a value. Its `.value` belongs in a downstream cell so Marimo can track the dependency. Predict the speed before moving the duration slider. The ordinary Python calculation is the same as before.
    """)
    return


@app.cell
def _(mo):
    duration_control = mo.ui.slider(1, 10, value=4, step=1, label="Time for 6 m (s)")
    duration_control
    return (duration_control,)

@app.cell
def _(duration_control, mo, walking_speed):
    mo.md(f"Walking speed: **{walking_speed(6, duration_control.value):.2f} m/s**")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 5.** The slider value changes but a separate scratch cell contains the literal calculation `6 / 4`. Will that expression change?

    /// details | Solution to Exercise 5
    No. It has no dependency on the slider. Use `6 / duration_control.value` in a downstream cell to connect the calculation to the input. This is the same distinction as using a named variable instead of copying its current value.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. What is the difference between assignment and comparison?
    2. Why keep a variable’s definition in one Marimo cell?
    3. Does an exception mean Python silently returned zero?
    4. What should every reported physical number include?

    /// details | Checkpoint answers
    1. `=` assigns; `==` compares.
    2. It gives the dependency graph one unambiguous source.
    3. No; the calculation failed.
    4. Units and enough context to interpret it.
    ///

    ## Problems

    1. A 12 m walk takes 8 s. Give its speed.
    2. For `values = [2, 4, 6, 8]`, give `values[1:3]` and `len(values)`.
    3. Give a Boolean expression that checks a duration is greater than zero and at most 10 s.
    4. State the output of `walking_speed(0, 3)` and the outcome of `walking_speed(3, -1)`.

    /// details | Final answers
    1. 1.5 m/s.
    2. `[4, 6]`; 4.
    3. `0 < duration_s <= 10`.
    4. 0 m/s; `ValueError`.
    ///

    ## Go deeper

    [Python tutorial](https://docs.python.org/3/tutorial/) — expressions, data structures, and functions. [Marimo key concepts](https://docs.marimo.io/getting_started/key_concepts/) — reactive cells. Continue with [Scalar and vector](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/scalar_vector.py) for NumPy arrays or [Code structure](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/code_structure.py) when you need repeated file processing.

    ### Video lectures on the Internet

    [MIT introductory Python lectures](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/video_galleries/lecture-videos/) — use the first lectures for expressions, variables, branching, and loops.

    ## References

    - Python Software Foundation, [Python 3 tutorial](https://docs.python.org/3/tutorial/). Marimo team, [Key concepts](https://docs.marimo.io/getting_started/key_concepts/).
    """)
    return


if __name__ == "__main__":
    app.run()
