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
    # Text and binary files

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A sensor sends the value 1.5. How can the same number appear as readable characters or as four binary bytes?

    We will distinguish an encoding from a file layout and inspect both representations in memory. No files or extra packages are required.

    Predict the result before running each example. Try the **Exercises** before opening their worked solutions. Finish with the **Checkpoint questions** and **Problems**; the latter provide final answers only.

    Open this file locally in marimo, on [marimo.app](https://marimo.app/), or in [molab](https://molab.marimo.io/notebooks). The examples are self-contained.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    Python's built-in text and byte types and its `struct` module are enough for these examples.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import struct
    return mo, struct


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All files contain bytes

    Text formats interpret bytes using a character encoding. ASCII defines 128 character codes; UTF-8 can encode Unicode characters, including accented letters. A file extension is a convention, not proof of an encoding. To interpret a scientific text table, we also need its delimiter, column meanings, and units.

    A binary format directly specifies numeric representation and structure. It may be more compact or faster to parse, but that depends on the data and format; neither is guaranteed merely by calling a file binary.
    """)
    return


@app.cell
def _():
    text_bytes = "1.5\n".encode("utf-8")
    text_bytes, len(text_bytes)
    return (text_bytes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** How many bytes does the UTF-8 text `1.5` occupy without a newline?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 1

    Three bytes: one for each ASCII character `1`, `.`, and `5`. A trailing newline adds one byte in this example.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A number needs a representation

    The format `<f` specifies a little-endian 32-bit floating-point value. Decoding the same bytes with the wrong type or byte order can produce a plausible-looking but incorrect number. This is why binary readers need a format specification.
    """)
    return


@app.cell
def _(struct):
    binary_bytes = struct.pack("<f", 1.5)
    binary_bytes, len(binary_bytes), struct.unpack("<f", binary_bytes)[0]
    return (binary_bytes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** Does seeing `1.5` in a text column tell you whether it is metres, seconds, or volts?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 2

    No. Character decoding recovers the text, but physical interpretation requires metadata or a documented column schema.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Is UTF-8 a numerical data schema?

    2. Does a .txt extension guarantee ASCII?

    3. Are text files stored as physical characters rather than bits?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Checkpoint answers

    1. No; it is a character encoding.
    2. No.
    3. No; both text and other formats are stored as bytes.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    Try these independently; open a panel only to check your final answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 1.** How many bytes are used by ten 32-bit floating-point values, excluding headers?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Answer to Problem 1

    40 bytes.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** What two numerical conventions does `<f` specify?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Answer to Problem 2

    Little-endian byte order and 32-bit floating-point representation.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Apply these ideas in the [C3D guide](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/open_c3d_file.py) and [Cortex text-file guide](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/read_mac_ascii_files.py).

    ### Video lectures on the Internet

    [Harvard CS50 lecture archive](https://cs50.harvard.edu/x/2026/weeks/0/) introduces binary representation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Python: struct](https://docs.python.org/3/library/struct.html).

    - [Python Unicode how-to](https://docs.python.org/3/howto/unicode.html).
    """)
    return


if __name__ == "__main__":
    app.run()
