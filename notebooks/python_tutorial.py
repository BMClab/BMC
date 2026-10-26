# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.9",
#     "matplotlib>=3.10",
#     "numpy>=2.0",
#     "pandas>=2.2",
#     "scipy>=1.14",
#     "sympy>=1.13",
# ]
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

    Learn Python by calculating, organizing measurements, and exploring scientific examples in a [marimo](https://marimo.io/) notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    This tutorial introduces the Python language and its scientific computing tools. No programming experience is assumed. We use marimo to put explanations, executable Python, figures, and experiments together. The Python you learn here also applies to programs outside a notebook.

    Work through the guide in stages:

    1. **Get oriented:** open the notebook and learn how marimo runs cells.
    2. **Learn Python:** expressions, names, types, collections, decisions, loops, functions, and errors.
    3. **Work with scientific data:** NumPy arrays, interpolation, files, Matplotlib figures, SciPy signal processing, SymPy mathematics, and pandas tables.
    4. **Put it together:** explore an interactive model and analyze a set of walking trials.

    Read the examples in order. Before running a cell, predict its result; afterward, change an input and explain what happened. Try each **Exercise** before opening its worked solution. The final **Problems** provide independent practice, with final answers in a separate panel. The scientific sections can be studied over several sessions.

    **Challenge 0.** A walker covers 6 metres in 4 seconds. How would you calculate the average speed? How would your program handle three trials, or a trial whose duration was recorded as zero?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    The language examples need Python and marimo. The scientific examples also use NumPy, Matplotlib, SciPy, SymPy, and pandas. This repository declares its environment in `pyproject.toml` and `uv.lock`; the notebook also declares its dependencies for services that open it independently.

    For installation instructions, see [Python installation](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/python_installation.py). From a local copy of the repository, enter these commands in a **terminal**, in the repository's root folder:

    ```bash
    uv sync
    uv run marimo edit notebooks/python_tutorial.py
    ```

    If you are reading the saved HTML page, choose **Edit in browser** or a launch badge to run and edit the examples. The saved page shows results from a previous execution; it cannot recalculate Python when you move a slider. Download the `.py` file to keep a copy of the notebook.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction to marimo

    A notebook is a document made of **cells**. Text cells explain a question or a result; code cells contain Python that you can edit and execute. marimo saves this document as a Python `.py` file, so you can download it, reopen it, and track its changes with version control.

    The official [marimo Getting Started guide](https://docs.marimo.io/getting_started/) includes installation, a quickstart, and a tour of the key concepts. Here is enough to work through this tutorial:

    1. Open the notebook in the editor and run its cells with **Run all** if they have not run already.
    2. Read a code cell, edit it, and use its run control to execute the edited code.
    3. The cell's **last expression** is displayed as its result. Use `print(...)` to display several intermediate values or messages. An assignment alone does not display the assigned value.
    4. Add a code cell for your experiments. Text cells use Markdown, commonly through `mo.md(...)`, to combine prose, links, and mathematics. The editor manages the surrounding cell definitions in the saved Python file.

    ### Reactive execution

    marimo tracks which cells define and use each name. In automatic execution mode, running a changed cell also reruns the cells that depend on it. In lazy execution mode, dependent cells are marked stale and you run them when ready. Dependencies determine execution order; reading the notebook from top to bottom gives the teaching order.

    Keep each shared name, such as `distance_m`, **defined in one cell**. To change an input, edit that defining cell. Ordinary reassignment within a cell or a function is still valid Python. For a scratch calculation, a name beginning with `_`, such as `_result`, is local to that marimo cell and can be reused in another cell. This cell-local behavior is a marimo convention, not a general Python scoping rule.

    Also avoid defining a list or table in one cell and modifying it in another: marimo does not track those mutations. Make the changes in the cell that creates the object, or create a new object with a new name. We will practice this when working with collections.

    **Guiding question.** If a result should change when a walking time changes, should its code use the name of that time or a copied number? Look for this dependency in the next examples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python as a calculator

    An **expression** combines values and operations to produce a value. Run this first calculation, then replace one of its numbers. The result is an average speed in metres per second because we supplied metres and seconds; Python itself does not attach units to ordinary numbers.
    """)
    return


@app.cell
def _():
    6 / 4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The main arithmetic operators are:

    | Operator | Meaning | Example |
    | --- | --- | --- |
    | `+`, `-` | Addition, subtraction | `6 + 4`, `6 - 4` |
    | `*`, `/` | Multiplication, division | `6 * 4`, `6 / 4` |
    | `//`, `%` | Floor division, remainder | `7 // 3`, `7 % 3` |
    | `**` | Exponentiation | `3 ** 2` |

    Multiplication and division take precedence over addition and subtraction. Parentheses make the intended order explicit. `**` is exponentiation; `^` is a different operation, called bitwise exclusive OR. `/` produces a floating-point result even when both operands are integers.
    """)
    return


@app.cell
def _():
    print("2 + 3 * 4 =", 2 + 3 * 4)
    print("(2 + 3) * 4 =", (2 + 3) * 4)
    print("7 / 3 =", 7 / 3)
    print("7 // 3 =", 7 // 3, "and 7 % 3 =", 7 % 3)
    print("3 ** 2 =", 3**2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Assignment, expressions, and names

    `=` assigns the value on its right to the name on its left. A variable name lets you refer to that value later. Use meaningful names, with units when useful, instead of relying on your memory of what a number means. Python names are case sensitive: `time_s` and `Time_s` are different names.

    The following cell defines two inputs. The next cell uses them to calculate a result. Change `time_s` from `4.0` to `8.0` in its defining cell and run that cell. In automatic mode, the speed updates too. Restore `4.0` before continuing if you want to reproduce the printed answers in this guide.
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
    Assignment is an instruction, not an equation to solve. `_count = _count + 1` evaluates the old value, adds one, and assigns the result back to the name. `_count += 1` is a shorter way to express this update for a number. Multiple assignment can unpack values or swap them.

    These scratch names begin with `_` so they stay in this marimo cell. The statements inside the cell execute in their written order.
    """)
    return


@app.cell
def _():
    _count = 2
    _count = _count + 1
    _count += 1
    _left, _right = 5, 6
    _left, _right = _right, _left
    print("Count:", _count)
    print("After swapping:", _left, _right)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** Predict `2 ** 3`, `8 / 2`, and `8 // 3`. For the walking example, what happens when time doubles? Would a separate cell containing `6 / 4` change too?

    /// details | Solution to Exercise 1
    The results are `8`, `4.0`, and `2`. Doubling time from 4 to 8 s halves the speed from 1.5 to 0.75 m/s. A literal expression `6 / 4` stays unchanged because it does not refer to `time_s`. Use the variable to express the dependency.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports, objects, and help

    Python provides built-in functions such as `print`, `len`, and `type`. Other functions are organized in **modules** and **packages**. The `import` statement makes a module available; it is a statement, not a function call. The standard library comes with Python. Scientific packages such as NumPy are installed separately in the environment.

    After `import math`, use a dot to access a name in that module: `math.sqrt(9)` calls its square-root function. The trigonometric functions in `math` use radians. You can also import a particular name, as in `from math import sqrt`, or give a module an alias, as we will do with `import numpy as np`. Prefer explicit imports over `from module import *`, which obscures where names come from.
    """)
    return


@app.cell
def _():
    import math
    import platform

    print("Python version:", platform.python_version())
    print("Square root of 9:", math.sqrt(9))
    print("Cosine of pi radians:", math.cos(math.pi))
    return (math,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Objects, attributes, and methods

    Values in Python are **objects**. An object has a type and may expose attributes and methods through the dot operator. An attribute holds information; a method is an operation attached to the object. For example, `math.pi` is an attribute of the module, and `"trial".upper()` calls a string method. Later, an array's `.shape` will describe its dimensions.

    Python supports object-oriented programming with classes, but you can use these objects before learning to define your own classes.

    ### Finding out how something works

    `help(object)` displays documentation; `dir(object)` lists available names. A function's documentation explains its parameters and returned result. The marimo editor also offers completion and documentation while you type. Standard Python calls work in these cells; IPython-specific syntax such as `math.sqrt?` or `%matplotlib inline` is unnecessary here.
    """)
    return


@app.cell
def _(math):
    help(math.sqrt)
    print("Available names in math:", dir(math))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comments and readable code

    A comment starts with `#` and continues to the end of the line. Explain an assumption or a reason, such as `# Ignore the first sample because acquisition had not started`. Notebook text explains the broader scientific argument. A **docstring** is a string at the start of a function, class, or module that documents it; a triple-quoted string elsewhere is still a string, not a special multiline-comment syntax.

    Use descriptive `snake_case` names, spaces around arithmetic operators, and four spaces for each indentation level. Avoid names such as `list`, `sum`, or `print` for your own variables because they would hide Python's built-ins.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Values and types

    A value's **type** determines the operations it supports. Python infers types from values; you do not need to declare a variable's type before using it.

    | Type | Example | Typical use |
    | --- | --- | --- |
    | `int` | `3` | Counts and indices |
    | `float` | `3.0`, `1.2e-3` | Measurements and numerical calculations |
    | `complex` | `1 + 2j` | Numbers with real and imaginary parts |
    | `str` | `"trial 1"` | Text |
    | `bool` | `True`, `False` | Conditions |
    | `NoneType` | `None` | An absent value or no returned result |

    `3.0` is a float even though its numerical value is a whole number. `None` is not zero and is not the string `"None"`. Use `type(value)` to inspect a type, and explicit conversions such as `float("4.2")` when text represents a number.
    """)
    return


@app.cell
def _():
    print("3:", type(3))
    print("3.0:", type(3.0))
    print("1 + 2j:", type(1 + 2j))
    print('"trial 1":', type("trial 1"))
    print("True:", type(True))
    print("None:", type(None))
    print("Converted duration:", float("4.2"))
    print("Truncation toward zero:", int(4.9))
    print("Complex parts:", (1 + 2j).real, (1 + 2j).imag)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Floating-point arithmetic

    Most decimal fractions cannot be represented exactly by a finite binary floating-point number. Small rounding differences are therefore expected in numerical calculations. Displaying fewer decimal places changes how a number is shown; it does not change the stored value.

    Use a tolerance when comparing calculated floats. `math.isclose(a, b)` compares them approximately. Near zero, choose a meaningful absolute tolerance with `abs_tol`, considering the units and accuracy of your problem. A computer tolerance is not a substitute for an experimental uncertainty estimate.
    """)
    return


@app.cell
def _(math):
    print("0.1 + 0.2:", 0.1 + 0.2)
    print("Exact comparison:", 0.1 + 0.2 == 0.3)
    print("Approximate comparison:", math.isclose(0.1 + 0.2, 0.3))
    print("Residual near zero:", math.isclose(0.1 + 0.2 - 0.3, 0, abs_tol=1e-12))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Strings

    Strings hold text. Single or double quotes delimit a string; choose the form that makes embedded quotation marks easy to read. `\n` represents a newline. `+` joins strings and `*` repeats them. Neither operator converts numeric text into numbers automatically: `"4.0" + "4.2"` produces `"4.04.2"`.

    Indexing starts at zero. A negative index counts from the end. A slice `[start:stop:step]` includes `start` and excludes `stop`; omitted bounds extend to the beginning or end. These ideas also apply to lists and tuples.
    """)
    return


@app.cell
def _(speed_m_s):
    _word = "Python"
    _label = "  participant P01  "
    _csv_line = "P01,4.2"
    _label_parts = ["P01", "trial", "01"]
    print("First and last:", _word[0], _word[-1])
    print("Slices:", _word[:2], _word[2:], _word[::2])
    print("Cleaned label:", _label.strip().lower())
    print("Split fields:", _csv_line.split(","))
    print("Joined label:", "-".join(_label_parts))
    print(f"Speed: {speed_m_s:.2f} m/s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An **f-string**, prefixed with `f`, inserts expressions between braces. In `{speed_m_s:.2f}`, `.2f` requests two digits after the decimal point. String methods such as `.strip()` and `.lower()` return new strings: strings are immutable, so their characters cannot be changed in place.

    **Exercise 2.** What are `"Python"[1:4]`, `len("Python")`, and `float("4.0") + float("4.2")`? Does `int(4.9)` round to the nearest integer?

    /// details | Solution to Exercise 2
    The results are `"yth"`, `6`, and `8.2`. `int(4.9)` gives `4`: conversion truncates toward zero. Use a rounding operation when rounding is intended.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Collections: lists, tuples, sets, and dictionaries

    ### Lists and slicing

    A list stores an ordered sequence of objects. It can hold measurements, labels, or other lists. `len` gives its length; `sum`, `min`, and `max` summarize numeric items. Predict the first item, last item, and slice below before running the cells.
    """)
    return


@app.cell
def _():
    trial_times_s = [4.0, 4.2, 3.8]
    print("All durations:", trial_times_s)
    print("First and last:", trial_times_s[0], trial_times_s[-1])
    print("Items at positions 1 and 2:", trial_times_s[1:3])
    print("Reversed:", trial_times_s[::-1])
    print(
        "Count:", len(trial_times_s), "Mean:", sum(trial_times_s) / len(trial_times_s)
    )
    return (trial_times_s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lists are **mutable**: you can replace an item or append one. Assignment does not copy an object. Two names can refer to the same list, so modifying it through either name changes that list. `.copy()` makes a shallow copy, which is enough for this flat list of numbers. A nested list needs more care because its inner objects are still shared.

    `.append(...)` changes the list and returns `None`; `sorted(...)` returns a new sorted list. All mutations in the next example happen in the cell that creates the list, keeping the marimo dependency graph clear.
    """)
    return


@app.cell
def _():
    _original = [4.0, 4.2, 3.8]
    _alias = _original
    _copy = _original.copy()
    _alias[0] = 5.0
    _original.append(4.1)
    print("Original and alias:", _original, _alias)
    print("Independent copy:", _copy)
    print("Sorted copy:", sorted(_original))
    return


@app.cell
def _(trial_times_s):
    extended_times_s = trial_times_s + [4.1]
    print("A new list:", extended_times_s)
    print("The input list:", trial_times_s)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Other useful containers

    | Container | Example | Main idea |
    | --- | --- | --- |
    | List | `[4.0, 4.2, 3.8]` | Ordered, mutable sequence |
    | Tuple | `(6.0, 4.0)` | Immutable sequence; convenient for fixed groups of values |
    | Set | `{"P01", "P02"}` | Distinct elements; no positional indexing |
    | Dictionary | `{"label": "P01", "distance_m": 6.0}` | Values accessed by keys |

    A tuple's items cannot be replaced, although a mutable object inside a tuple can still change. An empty set is `set()`; `{}` creates an empty dictionary. `in` tests membership: in a dictionary it checks keys. A dictionary preserves insertion order, but a set's iteration order should not be used to assign scientific meaning.
    """)
    return


@app.cell
def _():
    _measurement = (6.0, 4.0)
    _distance, _duration = _measurement
    _labels = ["P01", "P02", "P01"]
    _participants = set(_labels)
    participant = {"label": "P01", "distance_m": 6.0}
    print("Unpacked tuple:", _distance, _duration)
    print("Unique labels, sorted for display:", sorted(_participants))
    print("Dictionary lookup:", participant["label"])
    print("Dictionary items:", list(participant.items()))
    print("Has a distance?", "distance_m" in participant)
    return (participant,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** For `values = [2, 4, 6, 8]`, predict `values[1:3]` and `values[-1]`. If `other = values`, does `other.append(10)` leave `values` unchanged? Which container would associate a participant's label with their walking distance?

    /// details | Solution to Exercise 3
    The slice is `[4, 6]` and the last item is `8`. Both names refer to the same list, so appending through `other` also changes what `values` refers to. Use a dictionary to associate named keys with values. In marimo, keep the aliasing experiment in one scratch cell.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparisons and Boolean logic

    Comparisons such as `==`, `!=`, `<`, `<=`, `>`, and `>=` produce `True` or `False`. `=` assigns; `==` compares. Combine conditions with `and`, `or`, and `not`. Python allows chained comparisons such as `0 < duration_s <= 10`.

    For an absent value, write `value is None`. `is` tests object identity, whereas `==` tests equality of values. Empty containers and zero are false in conditions, but a scientific validity check should state its meaning explicitly: a negative duration is nonzero, yet physically invalid for our speed calculation.
    """)
    return


@app.cell
def _():
    _duration_s = 4.0
    _missing = None
    print("Positive and at most 10 s:", 0 < _duration_s <= 10)
    print("Outside that interval:", _duration_s <= 0 or _duration_s > 10)
    print("Missing:", _missing is None)
    print("Not positive:", not (_duration_s > 0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Control flow and indentation

    ### Decisions: `if`, `elif`, and `else`

    A program can choose a calculation according to a condition. A colon introduces a block; its indented statements belong together. Use four spaces for each level. `elif` checks another condition if the preceding one was false, and `else` handles the remaining cases. Only the first matching branch runs.

    Change `_duration_s` in the next cell to `4.0`, `0.0`, and `-1.0`. Predict which branch runs each time. The zero and negative cases should explain why a speed cannot be calculated.
    """)
    return


@app.cell
def _():
    _duration_s = 0.0
    if _duration_s > 0:
        print(f"Speed: {6.0 / _duration_s:.2f} m/s")
    elif _duration_s == 0:
        print("A zero duration would require division by zero.")
    else:
        print("A walking duration cannot be negative.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Repetition: `for`, `range`, `enumerate`, and `zip`

    A `for` loop visits each item of an iterable, such as a list or a string. `enumerate` supplies a counter along with each item. Its optional `start=1` makes printed trial labels begin at 1 without changing Python's zero-based indexing.
    """)
    return


@app.cell
def _(distance_m, trial_times_s):
    for _trial_number, _duration_s in enumerate(trial_times_s, start=1):
        print(f"Trial {_trial_number}: {distance_m / _duration_s:.3f} m/s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `range(start, stop, step)` represents a sequence of integers that excludes `stop`. It is not a list, but `list(range(...))` materializes its values. `zip` pairs items from iterables; `strict=True` reports unequal lengths instead of silently stopping at the shorter input.

    A **list comprehension** builds a list from an iteration and can include a condition. Read `[_distance / _time for _time in times]` as “calculate this expression for every time”. Prefer a loop when the processing needs several steps.
    """)
    return


@app.cell
def _(distance_m, trial_times_s):
    print("range(3):", list(range(3)))
    print("range(1, 7, 2):", list(range(1, 7, 2)))
    for _label, _duration in zip(["A", "B", "C"], trial_times_s, strict=True):
        print(_label, _duration)
    trial_speeds_m_s = [distance_m / _duration for _duration in trial_times_s]
    print("Speeds from a comprehension:", trial_speeds_m_s)
    return (trial_speeds_m_s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Repetition while a condition holds: `while`

    A `while` loop checks its condition before each iteration. Something in the loop must eventually make the condition false, or it may run indefinitely. Here we generate Fibonacci numbers, each the sum of the preceding two. Simultaneous assignment uses the old values on the right before updating either name on the left.

    `break` exits the nearest loop; `continue` skips to its next iteration. Use them when they make a stopping or exclusion rule clearer.
    """)
    return


@app.cell
def _():
    _previous, _current = 0, 1
    _fibonacci = []
    while _current < 100:
        _fibonacci.append(_current)
        _previous, _current = _current, _previous + _current
    print("Fibonacci numbers below 100:", _fibonacci)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.** Write a comprehension that keeps only positive durations from `[4.0, 0.0, 5.0, -1.0]`. How many iterations does `for _i in range(2, 8, 2)` perform? What would happen if the Fibonacci loop never updated `_current`?

    /// details | Solution to Exercise 4
    `[_time for _time in [4.0, 0.0, 5.0, -1.0] if _time > 0]` gives `[4.0, 5.0]`. The range supplies `2`, `4`, and `6`, giving three iterations. If the initially true `while` condition never changes, the loop does not terminate.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining and using functions

    A function names a reusable calculation. `def` introduces the function, parentheses contain its **parameters**, and the indented body runs when it is called. The values supplied in a call are its **arguments**. `return` sends a result back to the caller; `print` only displays a representation. A function without an explicit return value returns `None`.

    Write a docstring that explains the meaning, units, and valid range of the inputs and result. This version of our walking calculation rejects nonfinite numbers, negative distances, and nonpositive durations rather than reporting a plausible-looking speed.
    """)
    return


@app.cell
def _(math):
    def walking_speed(distance_m, duration_s):
        """Return speed in m/s for finite distance >= 0 and duration > 0.

        Raise ValueError when either measurement is outside that domain.
        """
        if not math.isfinite(distance_m) or not math.isfinite(duration_s):
            raise ValueError("Measurements must be finite")
        if distance_m < 0 or duration_s <= 0:
            raise ValueError("Use nonnegative distance and positive duration")
        return distance_m / duration_s

    def speed_report(speed_m_s, label="Trial"):
        """Return a labelled speed, formatted to two decimal places."""
        return f"{label}: {speed_m_s:.2f} m/s"

    return speed_report, walking_speed


@app.cell
def _(speed_report, walking_speed):
    _speed = walking_speed(10, 8)
    print(speed_report(_speed))
    print(speed_report(walking_speed(duration_s=5, distance_m=6), label="Second trial"))
    print("A returned value can be reused:", _speed * 3.6, "km/h")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The second call uses **keyword arguments**, making the inputs explicit. `label="Trial"` defines a default argument for `speed_report`. Avoid mutable defaults such as an empty list when you intend a new object on every call.

    Function parameters and variables created inside a function are local to that call. They can have the same names as notebook variables without redefining those shared variables. Pass the measurements into the function rather than hiding them as fixed numbers inside its body. In marimo, define the function in one cell and call it in downstream cells.

    **Exercise 5.** Write a function `metres_to_centimetres(length_m)` that returns the converted length. What do `walking_speed(0, 3)` and `walking_speed(3, 0)` do? Why would replacing `return` with `print` prevent later calculations from using a function's result?

    /// details | Solution to Exercise 5
    ```python
    def metres_to_centimetres(length_m):
        '''Return a length in centimetres from a length in metres.'''
        return 100 * length_m
    ```

    The first speed is `0.0` m/s. The zero-duration call raises `ValueError`. A function that only prints, with no return statement, returns `None`; the displayed text is not a numeric result that a caller can multiply or average.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding and handling errors

    An exception reports that an operation failed. Read the final error line for its type and message, then inspect the indicated line in the traceback. A syntax or indentation error prevents code from being parsed; runtime exceptions arise while executing valid syntax.

    | Exception | Example cause | First thing to check |
    | --- | --- | --- |
    | `NameError` | Using a name that has not been defined | Spelling and the defining cell |
    | `TypeError` | Adding a string to a number | Types and explicit conversion |
    | `IndexError` | Selecting item 3 from a three-item list | Zero-based indices and length |
    | `KeyError` | Looking up an absent dictionary key | Available keys |
    | `ZeroDivisionError` | Dividing by zero | Input validity |
    | `ValueError` | A function rejects an argument's value | The function's documented domain |

    `try` and `except` let a program handle an expected failure. Catch the specific exception you can handle. In the next cell, invalid trials are reported and the loop continues; they are not replaced by zero speeds. The notebook itself can still execute all its cells successfully.
    """)
    return


@app.cell
def _(walking_speed):
    for _duration in [4.0, 0.0, -1.0]:
        try:
            _speed = walking_speed(6.0, _duration)
        except ValueError as _error:
            print(f"Duration {_duration} s: {_error}")
        else:
            print(f"Duration {_duration} s: {_speed:.2f} m/s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical data with NumPy

    We now have enough Python to start using scientific libraries. NumPy provides **arrays**, which store numerical data with a shape and a data type and support operations on many elements at once. The conventional alias is `np`.

    Python lists and NumPy arrays have different arithmetic: multiplying a list by 2 repeats its items, whereas multiplying a numeric array by 2 doubles each value. Converting our trial times to an array lets one division calculate every speed.
    """)
    return


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _(distance_m, np, trial_times_s):
    times_array_s = np.array(trial_times_s)
    speeds_array_m_s = distance_m / times_array_s
    print("List * 2:", trial_times_s * 2)
    print("Array * 2:", times_array_s * 2)
    print("Speeds [m/s]:", speeds_array_m_s)
    print("Mean speed [m/s]:", speeds_array_m_s.mean())
    print("Speeds above 1.5 m/s:", speeds_array_m_s[speeds_array_m_s > 1.5])
    return speeds_array_m_s, times_array_s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Shape, indexing, and axes

    A one-dimensional array can hold one participant's repeated measurements. A two-dimensional array might put participants in rows and trials in columns. `.shape` gives the size of each dimension, `.ndim` the number of dimensions, `.size` the total number of elements, and `.dtype` the stored element type.

    With two dimensions, separate indices with a comma: `[0, 1]` selects row 0, column 1; `[:, 0]` selects column 0 from every row. In `mean(axis=0)`, the row dimension is reduced, leaving a mean for each column. `axis=1` reduces columns and leaves one mean per row. Sketch the following array before predicting these means.
    """)
    return


@app.cell
def _(np):
    measurement_array_s = np.array([[4.0, 4.2, 3.8], [4.5, 4.3, 4.4]])
    print("Durations [s]:\n", measurement_array_s)
    print("Shape:", measurement_array_s.shape, "Dimensions:", measurement_array_s.ndim)
    print(
        "Total elements:", measurement_array_s.size, "Type:", measurement_array_s.dtype
    )
    print("First trial for each participant:", measurement_array_s[:, 0])
    print("Mean for each trial:", measurement_array_s.mean(axis=0))
    print("Mean for each participant:", measurement_array_s.mean(axis=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Basic NumPy slices often return **views**, sharing the original array's data. Use `.copy()` when you need an independent array. As with lists, keep in-place changes in the defining cell. Use a new array expression for a downstream transformation.

    Array comparisons produce Boolean arrays, also called masks. Combine elementwise conditions with `&` and `|`, placing each comparison in parentheses: `(times_array_s > 0) & (times_array_s < 5)`. Python's `and` and `or` instead combine individual truth values.

    `np.arange` is convenient for regularly spaced values with an excluded endpoint; `np.linspace` chooses a requested number of samples between endpoints. `np.zeros`, `np.ones`, and `np.column_stack` create or assemble arrays. For random examples, use a seeded generator so rerunning the lesson reproduces the same data.
    """)
    return


@app.cell
def _(np):
    print("Every second integer:", np.arange(0, 7, 2))
    print("Five equally spaced times:", np.linspace(0, 1, 5))
    print("A 2 by 3 array of zeros:\n", np.zeros((2, 3)))
    _rng = np.random.default_rng(2026)
    print("Reproducible synthetic speeds:", _rng.normal(loc=1.5, scale=0.1, size=4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 6.** An array has two participant rows and three trial columns. What are its shape and size? Which axis gives one mean per participant? How does `[1, 2] * 2` differ from `np.array([1, 2]) * 2`?

    /// details | Solution to Exercise 6
    The shape is `(2, 3)` and the size is `6`. `mean(axis=1)` gives one mean per participant. The list becomes `[1, 2, 1, 2]`; the array becomes `[2, 4]`. See the [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html) for more array practice.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpolation

    Suppose position was measured at four times and we want estimates between them. `np.interp(new_times, measured_times, measured_positions)` performs one-dimensional linear interpolation. The measured time coordinates must be increasing. More interpolated points make a denser representation; they do not add observations or recover motion that was never measured.

    We keep the new times inside the observed interval. Outside it, `np.interp` uses endpoint values by default rather than extending the slope. We will plot the measured and interpolated positions in the Matplotlib section.
    """)
    return


@app.cell
def _(np):
    sample_time_s = np.array([0.0, 1.0, 2.0, 3.0])
    sample_position_m = np.array([0.0, 1.2, 2.0, 3.6])
    dense_time_s = np.linspace(sample_time_s[0], sample_time_s[-1], 13)
    interpolated_position_m = np.interp(dense_time_s, sample_time_s, sample_position_m)
    print(
        "Estimated position at 0.5 s:",
        np.interp(0.5, sample_time_s, sample_position_m),
        "m",
    )
    print("New positions [m]:", interpolated_position_m)
    return dense_time_s, interpolated_position_m, sample_position_m, sample_time_s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading and saving files

    Scientific programs need inputs and outputs beyond the notebook. Text files store characters using an encoding such as UTF-8; a CSV file is text organized into rows and separated fields. Binary formats store data according to a particular format rather than as readable numeric text. Match the reader to the format.

    `pathlib.Path` represents a path. `/` joins path components; it is not numerical division when the left operand is a `Path`. A relative path is resolved from the working directory, which may differ between local and hosted sessions. A `with` block closes an opened file even if an exception occurs. Mode `"w"` writes and **replaces existing contents**; mode `"r"` reads.

    These examples use temporary folders so repeated runs do not overwrite your data. The folder is removed when its `with` block finishes. To retain a result in your own work, choose a persistent output folder or download it from the service you are using.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    return Path, TemporaryDirectory


@app.cell
def _(Path, TemporaryDirectory):

    with TemporaryDirectory(prefix="bmc-python-text-") as _folder:
        _path = Path(_folder) / "notes.txt"
        with _path.open("w", encoding="utf-8") as _stream:
            _stream.write("Participant: P01\nDistance: 6 m\n")
        with _path.open("r", encoding="utf-8") as _stream:
            _contents = _stream.read()
        print(_contents)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    NumPy provides `savetxt` and `loadtxt` for rectangular numeric data. Here each row contains a time and a position. The header records names and units, the delimiter is a comma, and the reader skips the header row. For labelled tables with text and numbers, pandas will provide a more convenient CSV reader later.
    """)
    return


@app.cell
def _(Path, TemporaryDirectory, np, sample_position_m, sample_time_s):
    _data = np.column_stack((sample_time_s, sample_position_m))
    with TemporaryDirectory(prefix="bmc-python-array-") as _folder:
        _path = Path(_folder) / "positions.csv"
        np.savetxt(_path, _data, delimiter=",", header="time_s,position_m", comments="")
        _reloaded = np.loadtxt(_path, delimiter=",", skiprows=1)
        print("Read back:\n", _reloaded)
        print("Values preserved:", np.allclose(_data, _reloaded))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 7.** What position does linear interpolation predict halfway between `(1 s, 1.2 m)` and `(2 s, 2.0 m)`? Does saving 1,000 interpolated points turn four observations into 1,000 observations? Why use a `with` block for a file?

    /// details | Solution to Exercise 7
    The estimate at 1.5 s is 1.6 m. Interpolation does not add independent observations. The `with` block closes the file reliably, including when an operation raises an exception.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting with Matplotlib

    A figure helps us check a calculation and communicate its meaning. Matplotlib's `plt.subplots` returns a **Figure**, containing the whole drawing, and one or more **Axes**, where data are plotted. Call methods on an Axes to draw lines, label quantities with units, and add a legend. This also puts Python's objects and methods to practical use.

    We will compare measured points with their interpolated curve, then show a histogram of synthetic walking speeds. A histogram groups observations into bins; its appearance depends on the chosen bins. Synthetic data demonstrate code and are not evidence about real participants.

    In marimo, put the Figure object at the end of the cell to display it inline. No `%matplotlib` magic or separate desktop window is needed. When an input changes, create the updated figure in its dependent cell.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(dense_time_s, interpolated_position_m, np, plt, sample_position_m, sample_time_s):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    _axes[0].plot(
        dense_time_s, interpolated_position_m, "-", label="Linear interpolation"
    )
    _axes[0].plot(sample_time_s, sample_position_m, "o", label="Measured samples")
    _axes[0].set(
        xlabel="Time [s]", ylabel="Position [m]", title="Samples and interpolation"
    )
    _axes[0].legend()
    _axes[0].grid(alpha=0.3)

    _synthetic_speeds = np.random.default_rng(2026).normal(1.5, 0.12, size=100)
    _axes[1].hist(_synthetic_speeds, bins=12, edgecolor="white")
    _axes[1].set(
        xlabel="Speed [m/s]", ylabel="Count", title="100 synthetic walking speeds"
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 8.** Which line of code distinguishes the measured positions from the interpolated estimates? Change the histogram to six bins: did the observations change? For a figure of your own, what information belongs on each axis?

    /// details | Solution to Exercise 8
    The `plot` call with marker `"o"` shows the measured samples; the line shows the interpolation. Changing the bins regroups the same synthetic observations. Each axis needs the plotted quantity and, where applicable, its unit. See the [Matplotlib quick start](https://matplotlib.org/stable/users/explain/quick_start.html).
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Signal processing with SciPy

    SciPy builds on NumPy with algorithms for integration, optimization, interpolation, linear algebra, statistics, and signal processing. This introduction follows one example: separating a slow movement component from a faster oscillation.

    We sample a 2 Hz sine wave plus a smaller 20 Hz sine wave at 100 samples per second. A low-pass Butterworth filter attenuates the faster component. `butter(..., fs=...)` takes its cutoff in the same frequency unit as the sampling rate. The cutoff must be between zero and half the sampling rate. We request second-order sections with `output="sos"` and apply them using `sosfiltfilt`.

    This is **offline forward-backward filtering**: it uses future samples, has zero phase shift, and doubles the filter order. Edge effects still need inspection, and the combined response at the specified single-pass cutoff is not the single-pass response. A cutoff should follow the scientific question and sampling process, not just make a curve look smooth.
    """)
    return


@app.cell
def _(np):
    from scipy.fft import rfft, rfftfreq
    from scipy.signal import butter, sosfiltfilt

    sampling_rate_hz = 100.0
    signal_time_s = np.arange(200) / sampling_rate_hz
    raw_signal = np.sin(2 * np.pi * 2 * signal_time_s) + 0.25 * np.sin(
        2 * np.pi * 20 * signal_time_s
    )
    _sos = butter(4, 5.0, btype="lowpass", fs=sampling_rate_hz, output="sos")
    filtered_signal = sosfiltfilt(_sos, raw_signal)
    return filtered_signal, raw_signal, rfft, rfftfreq, sampling_rate_hz, signal_time_s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `scipy.fft.rfft` transforms a real-valued record into frequency components; `rfftfreq` supplies their frequencies. Here we plot the unnormalized FFT magnitudes to compare the same-length records. Peaks near 2 and 20 Hz identify the generated oscillations. Boundaries and a record containing incomplete cycles can spread energy across frequency bins.

    The bottom panels also show a finite-difference approximation to the **second derivative**. Dividing by the sample interval squared is equivalent to multiplying by the sampling rate squared. Differentiation emphasizes faster oscillations: a small fast component can dominate the derivative. Compare the raw and filtered results before choosing how to estimate movement derivatives.
    """)
    return


@app.cell
def _(
    filtered_signal,
    np,
    plt,
    raw_signal,
    rfft,
    rfftfreq,
    sampling_rate_hz,
    signal_time_s,
):
    _raw_d2 = np.diff(raw_signal, n=2) * sampling_rate_hz**2
    _filtered_d2 = np.diff(filtered_signal, n=2) * sampling_rate_hz**2
    _frequencies = rfftfreq(raw_signal.size, d=1 / sampling_rate_hz)
    _derivative_frequencies = rfftfreq(_raw_d2.size, d=1 / sampling_rate_hz)
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")

    _axes[0, 0].plot(signal_time_s, raw_signal, label="Raw")
    _axes[0, 0].plot(signal_time_s, filtered_signal, label="Filtered")
    _axes[0, 0].set(xlabel="Time [s]", ylabel="Signal [a.u.]", title="Time domain")
    _axes[0, 1].plot(_frequencies, np.abs(rfft(raw_signal)), label="Raw")
    _axes[0, 1].plot(_frequencies, np.abs(rfft(filtered_signal)), label="Filtered")
    _axes[0, 1].set(
        xlabel="Frequency [Hz]",
        ylabel="FFT magnitude [a.u.]",
        xlim=(0, 30),
        title="Frequency domain",
    )
    _axes[1, 0].plot(signal_time_s[1:-1], _raw_d2, label="Raw")
    _axes[1, 0].plot(signal_time_s[1:-1], _filtered_d2, label="Filtered")
    _axes[1, 0].set(xlabel="Time [s]", ylabel="Second derivative [a.u./s²]")
    _axes[1, 1].plot(_derivative_frequencies, np.abs(rfft(_raw_d2)), label="Raw")
    _axes[1, 1].plot(
        _derivative_frequencies, np.abs(rfft(_filtered_d2)), label="Filtered"
    )
    _axes[1, 1].set(
        xlabel="Frequency [Hz]",
        ylabel="Derivative FFT magnitude [a.u./s²]",
        xlim=(0, 30),
    )
    for _ax in _axes.flat:
        _ax.legend()
        _ax.grid(alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 9.** Which generated component should be attenuated by a 5 Hz low-pass filter? Could this same example use a cutoff of 60 Hz with a 100 Hz sampling rate? Is the forward-backward result suitable for an application that must respond before future measurements exist?

    /// details | Solution to Exercise 9
    The 20 Hz component should be attenuated much more than the 2 Hz component. A 60 Hz cutoff is outside the allowed interval below the 50 Hz Nyquist frequency. Forward-backward filtering requires future data, so this offline method cannot directly supply a real-time response. See the [SciPy signal-processing tutorial](https://docs.scipy.org/doc/scipy/tutorial/signal.html).
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Symbolic mathematics with SymPy

    NumPy and SciPy operate mainly on numerical values. SymPy can manipulate mathematical expressions containing symbols: solve equations, differentiate, integrate, and work with exact matrices. We use the conventional alias `sym`.

    Defining `symbolic_x` as a symbol does not assign a numerical value to it. We construct the polynomial $x^2-2x-3$, solve for its zeros, and calculate its derivative and a definite integral. `sym.latex` converts expressions to mathematical notation for a marimo text cell.
    """)
    return


@app.cell
def _():
    import sympy as sym

    return (sym,)


@app.cell
def _(mo, sym):
    symbolic_x = sym.symbols("x", real=True)
    polynomial = symbolic_x**2 - 2 * symbolic_x - 3
    polynomial_roots = sym.solve(polynomial, symbolic_x)
    polynomial_derivative = sym.diff(polynomial, symbolic_x)
    polynomial_integral = sym.integrate(polynomial, (symbolic_x, 0, 1))
    mo.md(rf"""
    Polynomial: $p(x) = {sym.latex(polynomial)}$

    Roots: ${sym.latex(polynomial_roots)}$

    Derivative: $p'(x) = {sym.latex(polynomial_derivative)}$

    Definite integral: $\int_0^1 p(x)\,dx = {sym.latex(polynomial_integral)}$
    """)
    return (
        polynomial,
        polynomial_derivative,
        polynomial_integral,
        polynomial_roots,
        symbolic_x,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### An exact rotation matrix

    A rotation about the $x$ axis illustrates symbolic matrices. `.subs(...)` substitutes a value for a symbol, and `*` multiplies SymPy matrices. Using `sym.pi / 2` preserves an exact angle; using a floating approximation to $\pi$ instead can produce small residuals. A vector along the positive $y$ axis becomes a vector along the positive $z$ axis after this right-handed 90-degree rotation.

    The same approach can construct and combine rotations about all three axes; the order of matrix multiplication matters. See [Three-dimensional transformations](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py) for the mechanics.
    """)
    return


@app.cell
def _(mo, sym):
    _theta = sym.symbols("theta", real=True)
    _rotation_x = sym.Matrix(
        [
            [1, 0, 0],
            [0, sym.cos(_theta), -sym.sin(_theta)],
            [0, sym.sin(_theta), sym.cos(_theta)],
        ]
    )
    _rotated = _rotation_x.subs(_theta, sym.pi / 2) * sym.Matrix([0, 1, 0])
    mo.md(rf"""
    $$R_x(\theta) = {sym.latex(_rotation_x)}$$

    $$R_x(\pi/2)\begin{{bmatrix}}0\\1\\0\end{{bmatrix}} = {sym.latex(_rotated)}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 10.** Check the polynomial's roots by substituting each into $x^2-2x-3$. What is the derivative at $x=2$? What distinguishes `sym.pi` from `math.pi`?

    /// details | Solution to Exercise 10
    Both $x=-1$ and $x=3$ give zero. The derivative is $2x-2$, so its value at 2 is 2. `sym.pi` represents the exact symbolic constant; `math.pi` is a floating-point approximation. See the [SymPy introductory tutorial](https://docs.sympy.org/latest/tutorials/intro-tutorial/index.html).
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data analysis with pandas

    A pandas **DataFrame** organizes observations in rows and named variables in columns. Different columns can contain different types, such as participant labels and numeric measurements. A column is a **Series**. We use the conventional alias `pd`.

    The following small table contains repeated walking trials under two conditions. We derive a speed column in the same cell that creates the table. Subsequent cells select and summarize it without modifying that shared object.
    """)
    return


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    walking_trials = pd.DataFrame(
        {
            "participant": ["P01", "P01", "P01", "P02", "P02", "P02"],
            "condition": [
                "comfortable",
                "comfortable",
                "fast",
                "comfortable",
                "comfortable",
                "fast",
            ],
            "distance_m": [6.0] * 6,
            "duration_s": [4.0, 4.2, 3.0, 4.5, 4.3, 3.2],
        }
    )
    walking_trials["speed_m_s"] = (
        walking_trials["distance_m"] / walking_trials["duration_s"]
    )
    walking_trials
    return (walking_trials,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `table["column"]` selects a column. `.loc[row_condition, columns]` selects rows by a condition and columns by their labels. `.iloc` instead uses integer positions. Here the equality comparison creates a mask selecting the fast condition.
    """)
    return


@app.cell
def _(walking_trials):
    walking_trials.loc[
        walking_trials["condition"] == "fast", ["participant", "speed_m_s"]
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `groupby` organizes observations by labels before applying a summary. `agg` requests named aggregations. We count and average the speeds by condition. `describe()` gives a broader descriptive summary of numeric columns.

    These are summaries of the listed **trials**. Participants contributed repeated observations, so treating all rows as independent participants would be a separate and often incorrect statistical assumption.
    """)
    return


@app.cell
def _(walking_trials):
    condition_summary = walking_trials.groupby("condition")["speed_m_s"].agg(
        ["count", "mean"]
    )
    condition_summary
    return (condition_summary,)


@app.cell
def _(walking_trials):
    walking_trials.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CSV tables

    `to_csv(..., index=False)` writes column names and data without adding the DataFrame's row index as another column. `read_csv` reads the table back and infers column types. For real data, check those types, missing values, decimal conventions, and units rather than assuming a successful read proves the data are correct.
    """)
    return


@app.cell
def _(Path, TemporaryDirectory, pd, walking_trials):
    with TemporaryDirectory(prefix="bmc-python-table-") as _folder:
        _path = Path(_folder) / "walking_trials.csv"
        walking_trials.to_csv(_path, index=False)
        _reloaded_table = pd.read_csv(_path)
        print("CSV columns:", list(_reloaded_table.columns))
        print("Rows read:", len(_reloaded_table))
    _reloaded_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 11.** Select the `duration_s` column for participant P01. How many fast trials are in the table? Does the grouped count tell you how many distinct participants took part?

    /// details | Solution to Exercise 11
    Use `walking_trials.loc[walking_trials["participant"] == "P01", "duration_s"]`, giving 4.0, 4.2, and 3.0 s. There are two fast trials. A row count counts trials, not distinct participants; use the participant identifiers to establish the latter. See [Getting started with pandas](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html).
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive exploration in marimo

    We can now combine a Python function, NumPy arrays, a Matplotlib figure, and marimo controls to explore a model. Consider a response that approaches a limiting amplitude:

    $$y(t) = A\left(1-e^{-t/\tau}\right).$$

    $A$ sets the limiting amplitude and $\tau>0$ sets how quickly the response approaches it. At $t=\tau$, the response is approximately $0.632A$. This is a mathematical example, not a fitted physiological model.

    First define the function. Then create two sliders and display them. Read each slider's `.value` in a **downstream cell**, where the curve and figure are calculated. Changing a control reruns that dependent cell. This replaces the need for notebook-specific plotting magics or manually updating existing plot lines.

    **Guiding questions.** Which control should change the curve's plateau? Which should change the time taken to reach a fixed fraction of that plateau? Predict the answer before moving either slider.
    """)
    return


@app.cell
def _(np):
    def saturating_response(time_s, amplitude, tau_s):
        """Return A * (1 - exp(-t/tau)) for a positive time constant in seconds."""
        if tau_s <= 0:
            raise ValueError("The time constant must be positive")
        return amplitude * (1 - np.exp(-time_s / tau_s))

    return (saturating_response,)


@app.cell
def _(mo):
    amplitude_control = mo.ui.slider(
        start=0, stop=10, step=0.5, value=5, label="Amplitude A", show_value=True
    )
    time_constant_control = mo.ui.slider(
        start=5,
        stop=100,
        step=5,
        value=50,
        label="Time constant tau [s]",
        show_value=True,
    )
    mo.hstack([amplitude_control, time_constant_control], wrap=True)
    return amplitude_control, time_constant_control


@app.cell
def _(amplitude_control, mo, np, plt, saturating_response, time_constant_control):
    _time = np.linspace(0, 300, 301)
    _amplitude = amplitude_control.value
    _tau = time_constant_control.value
    _response = saturating_response(_time, _amplitude, _tau)
    _fig, _ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    _ax.plot(_time, _response, label="Response")
    _ax.axhline(_amplitude, color="gray", linestyle="--", label="Limiting amplitude")
    _ax.set(xlabel="Time [s]", ylabel="Response [a.u.]", ylim=(-0.2, 10.5))
    _ax.grid(alpha=0.3)
    _ax.legend()
    mo.vstack([mo.md(f"**Current model:** A = {_amplitude:g}, tau = {_tau:g} s"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 12.** Double $A$ while holding $\tau$ fixed, then double $\tau$ while holding $A$ fixed. What changes in each experiment? Would a cell containing only `5 * (1 - np.exp(-10 / 50))` respond to these sliders?

    /// details | Solution to Exercise 12
    Doubling $A$ doubles the response at every time, including the limiting amplitude. Doubling $\tau$ makes the response take twice as long to reach the same fraction of its limit. The literal expression contains no reference to either control and therefore does not update when the sliders move. See [marimo interactive elements](https://docs.marimo.io/guides/interactivity/).
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Worked example: summarize walking trials

    Return to Challenge 0. We have four recorded durations, including a zero that cannot define a speed. Our goal is to calculate valid speeds, retain the reason for an excluded trial, and report a summary with units.

    Before running the code, identify the responsibilities: the function validates one calculation; the loop processes all records; a dictionary gives each result named fields; and pandas displays the table. Keeping the rejected trial in the table makes the exclusion visible.
    """)
    return


@app.cell
def _():
    recorded_durations_s = [4.0, 4.2, 0.0, 3.8]
    return (recorded_durations_s,)


@app.cell
def _(pd, recorded_durations_s, walking_speed):
    _records = []
    for _number, _duration in enumerate(recorded_durations_s, start=1):
        try:
            _speed = walking_speed(6.0, _duration)
        except ValueError as _error:
            _records.append(
                {
                    "trial": _number,
                    "duration_s": _duration,
                    "speed_m_s": None,
                    "status": str(_error),
                }
            )
        else:
            _records.append(
                {
                    "trial": _number,
                    "duration_s": _duration,
                    "speed_m_s": _speed,
                    "status": "valid",
                }
            )
    trial_results = pd.DataFrame(_records)
    trial_results
    return (trial_results,)


@app.cell
def _(mo, trial_results):
    _valid = trial_results.loc[trial_results["status"] == "valid", "speed_m_s"]
    _excluded_count = len(trial_results) - len(_valid)
    mo.md(
        f"**Summary:** {len(_valid)} valid trials; {_excluded_count} excluded trial(s). "
        + (
            f"Mean of valid trial speeds: **{_valid.mean():.3f} m/s**."
            if len(_valid)
            else "No valid speeds to average."
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The missing speed is displayed as `NaN` in the numeric column; it is not a measured speed of zero. The summary explicitly selects valid records. The mean of individual trial speeds is also a different quantity from total distance divided by total time. For three equal-distance trials with unequal durations, those summaries need not agree.

    **Try it.** Change one duration in `recorded_durations_s`, run its defining cell, and explain the updated table and mean. Then try making every duration zero: the summary should report no valid speeds rather than invent an average.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. How do assignment, equality comparison, and approximate floating-point comparison differ?
    2. What is the difference between a list and a NumPy array when multiplying by a scalar?
    3. Why can modifying an alias also change the original list?
    4. What do a function's parameters, docstring, and return value tell you?
    5. How do you read an error without hiding invalid data?
    6. What do rows and columns mean in your array or table?
    7. Why does a marimo cell update when an input changes, and what changes are not tracked?
    8. What must accompany a scientific number or figure so another person can interpret it?

    /// details | Checkpoint answers
    1. `=` assigns a value, `==` compares values, and a tolerance-based function such as `math.isclose` checks approximate agreement.
    2. A list repeats its items; a numeric array multiplies each element.
    3. The names refer to the same mutable object; assignment did not copy it.
    4. Parameters state the inputs, the docstring explains their meaning and domain, and the return value is the result the caller can reuse.
    5. Inspect the exception type, message, and indicated source line; validate inputs and handle only expected failures, recording exclusions.
    6. Their meaning comes from the data model, not the container; state whether they represent participants, trials, time points, or variables.
    7. marimo tracks dependencies through names. In-place mutations in another cell are not tracked; create a new object or mutate in its defining cell.
    8. Units, labels, input provenance, assumptions, and enough information to reproduce the calculation.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    Use new cells with distinct shared names, or scratch names beginning with `_`. Keep each problem's inputs explicit so another reader can reproduce the result.

    1. Calculate the speed of a 12 m walk completed in 8 s. Report it in m/s and km/h.
    2. For `[2, 4, 6, 8]`, give the slice selecting the middle two values, then calculate the mean using built-in functions.
    3. Write a Boolean expression that accepts durations greater than zero and at most 10 s.
    4. Write a function that takes a list of positive durations and a distance and returns a list of speeds. Use 6 m and `[4.0, 5.0, 3.0]` to check it.
    5. Explain how to keep an independent copy before changing the first element of a flat list. Then demonstrate it in one cell.
    6. Create the array `[[1, 2, 3], [4, 5, 6]]`. Find its shape and the mean of each row.
    7. Interpolate position at 1.5 s from times `[0, 1, 2]` s and positions `[0, 2, 6]` m. Plot the measured points and interpolated line with units.
    8. At a sampling rate of 100 Hz, which of the cutoffs 5, 40, and 60 Hz can be passed to the low-pass design used here? State whether that alone establishes a scientifically appropriate cutoff.
    9. Use SymPy to differentiate $x^3$ and integrate it from 0 to 2.
    10. For the response model, calculate the response at $t=\tau$ with $A=5$. Explain what happens when $\tau$ doubles.
    11. In the worked walking example, calculate both the mean of valid trial speeds and total valid distance divided by total valid time. Explain why they differ.
    12. Recreate the walking table from ordinary Python values, save it to a temporary CSV file, and read it back. Check the column names, units, and number of rows before summarizing by condition.

    /// details | Final answers
    1. 1.5 m/s; 5.4 km/h.
    2. `[1:3]` gives `[4, 6]`; the mean is 5.0.
    3. `0 < duration_s <= 10`.
    4. `[1.5, 1.2, 2.0]` m/s.
    5. Use `.copy()`; modifying the new flat list leaves the original unchanged.
    6. Shape `(2, 3)`; row means `[2.0, 5.0]`.
    7. 4.0 m.
    8. 5 and 40 Hz meet the numerical cutoff constraint; suitability still depends on the signal and scientific question.
    9. Derivative $3x^2$; definite integral 4.
    10. Approximately 3.161. Doubling $\tau$ doubles the time needed to reach that same response fraction.
    11. Approximately 1.502506 m/s and 1.5 m/s, respectively; the first averages ratios, while the second is a time-weighted average of speeds.
    12. Six rows and five named columns: `participant`, `condition`, `distance_m`, `duration_s`, and `speed_m_s`. Mean speeds are approximately 1.414 and 1.938 m/s for the comfortable and fast conditions.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Continue with [Scalar and vector](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/scalar_vector.py) for array-based mechanics, [Code structure](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/code_structure.py) for repeated file processing, and [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py) for a closer look at signal processing.

    For longer Python treatments, see Jake VanderPlas's [A Whirlwind Tour of Python](https://jakevdp.github.io/WhirlwindTourOfPython/) and [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/). Their Python examples complement this guide; adapt notebook-environment instructions when working in marimo.

    ### Video lectures on the Internet

    [MIT introductory Python lectures](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/video_galleries/lecture-videos/) offer additional practice with expressions, variables, branching, functions, and loops.

    ## References

    - Python Software Foundation, [Python 3 tutorial](https://docs.python.org/3/tutorial/) and [Floating-point arithmetic](https://docs.python.org/3/tutorial/floatingpoint.html).
    - marimo team, [Getting Started](https://docs.marimo.io/getting_started/), [Key concepts](https://docs.marimo.io/getting_started/key_concepts/), and [How marimo runs cells](https://docs.marimo.io/guides/reactivity/).
    - NumPy developers, [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html).
    - Matplotlib developers, [Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html).
    - SciPy developers, [Signal processing](https://docs.scipy.org/doc/scipy/tutorial/signal.html) and [Fourier transforms](https://docs.scipy.org/doc/scipy/tutorial/fft.html).
    - SymPy developers, [Introductory tutorial](https://docs.sympy.org/latest/tutorials/intro-tutorial/index.html).
    - pandas developers, [Getting started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html).
    """)
    return


if __name__ == "__main__":
    app.run()
