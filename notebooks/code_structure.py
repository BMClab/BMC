# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Code structure for data analysis

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A folder with twelve files is manageable by hand. A folder with twelve thousand is an invitation to lose a trial. Here we turn subject, condition, and repetition into explicit indices, then keep the same indices when a file is absent. You need Python loops and NumPy arrays.

    Read the examples in order, predict each result, and then run the cell. Try the distributed Exercises before opening their worked solutions. Finish with the checkpoint questions and independent Problems; their panels contain final answers only.

    **Challenge 0.** Two subjects each complete two conditions and three trials. If one file is missing, should the result have eleven rows or twelve? What information would either choice preserve?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy creates small synthetic text files and summarizes their columns. All demonstration files are written to a temporary directory, not to the repository data folder. This lesson intentionally deletes one demonstration file to show missing-data handling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sometimes data from experiments are stored in different files where each file contains data for different subjects, trials, conditions, etc. This text presents a common and simple solution to write a code to analyze such data.<br>
    The basic idea is that the name of the file is created in a structured way and you can use that to run a sequence of procedures inside one or more nested loops.<br>
    For instance, consider that the two first letters of the filename encode the initials of the subject's name, the next two letters the different conditions, and the last two characters the trial number.
    """)
    return


@app.cell
def _():
    subjects   = ['AA', 'AB']
    conditions = ['c1', 'c2']
    trials     = ['01', '02', '03']
    return conditions, subjects, trials


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could open and process these files with:
    """)
    return


@app.cell
def _(conditions, subjects, trials):
    for _subject in subjects:
        for _condition in conditions:
            for _trial in trials:
                _filename = _subject + _condition + _trial
                print(_filename)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The problem with this code is that if one one more files are missing or corrupted (which is typical), it will break. A solution is to read the file inside a `try` statement. The `try...except` handles exceptions such as a failure in reading a file and then we can use a `continue` statement to skip each failed iteration in the inner loop.<br>
    Let's create some files and implement this idea.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** Decode `ABc203.txt`. How many filenames do the nested loops generate, and which index changes fastest?

    /// details | Solution to Exercise 1
    The subject is AB, condition c2, trial 03. There are $2\times2\times3=12$ combinations. The innermost trial loop changes fastest; this order determines how we flatten results later.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read and save files

    If the data is in text ([ASCII](https://en.wikipedia.org/wiki/ASCII)) format, it's easier to read the file with the [`Numpy`](http://www.numpy.org/) function [`loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html) or with the [`pandas`](http://pandas.pydata.org/) function [`read_csv`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.io.parsers.read_csv.html). Both functions behave similarly; they can skip a certain number of first rows, can read files with different column separators, read numbers and letters, etc. `read_csv` tends to be faster but it returns a `pandas` [`DataFrame`](http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.html) object, which might not be useful if you are not into `pandas` (but you should be).

    To save data to a file, we can use the counterpart functions `savetxt` and `to_csv`:
    """)
    return


@app.cell
def _(conditions, subjects, trials):
    import numpy as np
    import tempfile as _tempfile
    from pathlib import Path as _Path
    path = _tempfile.mkdtemp(prefix='BMC-CodeStructure-') + '/'
    extension = '.txt'
    for _subject in subjects:
        for _condition in conditions:
            for _trial in trials:
                _filename = path + _subject + _condition + _trial + extension
                _data = np.random.randn(5, 3)
                _header = 'Col A\tCol B\tCol C'
                np.savetxt(_filename, _data, fmt='%g', delimiter='\t', header=_header, comments='')
                print('File', _filename, 'saved')
    return extension, np, path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The files are saved in a temporary folder, so that the notebook also runs where it cannot write next to itself (e.g., on molab). To save them somewhere else, change `path`; for instance, './../data/' moves up one directory relatively to the current directory (see the <a href="https://en.wikipedia.org/wiki/Cd_(command)">cd (command)</a>) and then into the folder data.<br>
    Let's remove one of the files:
    """)
    return


@app.cell
def _(path):
    from pathlib import Path
    missing_trial = Path(path) / 'AAc202.txt'
    missing_trial.unlink(missing_ok=True)
    return (missing_trial,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's read the data in these files and handle a possible missing or corrupted file:
    """)
    return


@app.cell
def _(conditions, extension, missing_trial, np, path, subjects, trials):
    assert not missing_trial.exists(), "Run the missing-trial demonstration first"
    for _subject in subjects:
        for _condition in conditions:
            for _trial in trials:
                _filename = path + _subject + _condition + _trial + extension
                try:
                    _data = np.loadtxt(_filename, skiprows=1)
                except (OSError, ValueError) as err:
                    print(_filename, err)
                    continue
                else:
                    print(_filename, 'loaded')
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** The missing trial is AAc202. Why should its mean remain NaN instead of zero? Which errors should the file-reading block catch?

    /// details | Solution to Exercise 2
    Zero is a possible measured mean; NaN marks unavailable information. Catch `OSError` for missing or unreadable files and `ValueError` for malformed numeric text. Catching every exception would also hide programming defects. Record the filename and error so a missing result remains traceable.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Store results

    The results of the analysis for each file can be stored in a variable in different ways.<br>
    We can store the results in a multidimensional variable where each dimension corresponds to the different indices in the loops. With the data above this would produce `results(s, c, t)`, a 2x2x3 array. Or we can store everything in a two-dimensional array where for example each row corresponds to each combination of subject, condition, and trial.<br>
    Let's try both ways:
    """)
    return


@app.cell
def _(conditions, extension, missing_trial, np, path, subjects, trials):
    assert not missing_trial.exists(), "Run the missing-trial demonstration first"
    results = np.empty(shape=(2, 2, 3, 3)) * np.nan
    for (_s, _subject) in enumerate(subjects):
        for (_c, _condition) in enumerate(conditions):
            for (_t, _trial) in enumerate(trials):
                _filename = path + _subject + _condition + _trial + extension
                try:
                    _data = np.loadtxt(_filename, skiprows=1)
                except (OSError, ValueError) as err:
                    continue
                else:
                    pass
                results[_s, _c, _t, :] = np.mean(_data, axis=0)
    print(results.shape)
    print(results)
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One problem with this approach is that for many dimensions the data gets convoluted and it might be difficult to read it.<br>
    The results for the first subject, condition, and trial are:
    """)
    return


@app.cell
def _(results):
    results[0, 0, 0, :]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use the second approach and store the results in a two-dimensional array:
    """)
    return


@app.cell
def _(conditions, extension, missing_trial, np, path, subjects, trials):
    assert not missing_trial.exists(), "Run the missing-trial demonstration first"
    results_1 = np.empty(shape=(2 * 2 * 3, 3)) * np.nan
    results2 = np.empty(shape=(2 * 2 * 3, 3)) * np.nan
    _ind = 0
    for (_s, _subject) in enumerate(subjects):
        for (_c, _condition) in enumerate(conditions):
            for (_t, _trial) in enumerate(trials):
                _ind = _ind + 1
                _filename = path + _subject + _condition + _trial + extension
                try:
                    _data = np.loadtxt(_filename, skiprows=1)
                except (OSError, ValueError) as err:
                    continue
                else:
                    pass
                results_1[_ind - 1, :] = np.mean(_data, axis=0)
                results2[len(conditions) * len(trials) * _s + len(trials) * _c + _t, :] = np.mean(_data, axis=0)
    print(results_1.shape)
    print(results_1)
    print(results2.shape)
    print(results2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can create columns identifying the subject, condition, and trial, which might be useful for running statistical analysis:
    """)
    return


@app.cell
def _(conditions, extension, missing_trial, np, path, subjects, trials):
    assert not missing_trial.exists(), "Run the missing-trial demonstration first"
    results_2 = np.empty(shape=(2 * 2 * 3, 3)) * np.nan
    _ind = 0
    indexes = []
    for (_s, _subject) in enumerate(subjects):
        for (_c, _condition) in enumerate(conditions):
            for (_t, _trial) in enumerate(trials):
                _ind = _ind + 1
                indexes.append([_s, _c, _t])
                _filename = path + _subject + _condition + _trial + extension
                try:
                    _data = np.loadtxt(_filename, skiprows=1)
                except (OSError, ValueError) as err:
                    continue
                else:
                    pass
                results_2[_ind - 1, :] = np.mean(_data, axis=0)
    results_2 = np.hstack((np.array(indexes), results_2))
    print(results_2.shape)
    print(results_2)
    return (results_2,)




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** For zero-based indices subject 1, condition 0, trial 2, find the row in the flattened 12-row result. Why keep identifier columns as well?

    /// details | Solution to Exercise 3
    The row is $2\times3\times1+3\times0+2=8$. Identifier columns preserve the experimental meaning if rows are subsequently sorted, filtered, or merged; an array position alone does not.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These are just some possible generic approaches to analyze data in multiple files.

    And we can save the results in a file:
    """)
    return


@app.cell
def _(np, path, results_2):
    _filename = path + 'results.txt'
    _header = 'Subject\tCondition\tTrial\tCol A\tCol B\tCol C'
    np.savetxt(_filename, results_2, fmt='%d\t%d\t%d\t%g\t%g\t%g', delimiter='\t', header=_header, comments='')
    print('File', _filename, 'saved')
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Which dimension of `results` represents measured columns?
    2. Why is `continue` inside the innermost loop?
    3. Does a successful file read prove the measurement is valid?

    /// details | Checkpoint answers
    1. The last dimension, of length three.
    2. It skips only the failed trial.
    3. No; check shape, units, range, and acquisition metadata too.
    ///

    ## Problems

    1. Four subjects, three conditions, five trials, and two measured columns: give both result shapes.
    2. One failed trial and one genuine zero mean occur in the same column. State how each should be represented.
    3. Give a reproducible record to accompany the exported result.

    /// details | Final answers
    1. `(4, 3, 5, 2)` and `(60, 2)`; with identifier columns, `(60, 5)`.
    2. NaN for unavailable data; 0 for the measured zero.
    3. Subject/condition/trial identifiers, units, source filenames, exclusion reasons, processing settings, and code version.
    ///

    ## Go deeper

    [Python tutorial: errors and exceptions](https://docs.python.org/3/tutorial/errors.html) — distinguish expected input failures from bugs. Follow with [Python tutorial](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/python_tutorial.py) for arrays and files.

    ### Video lectures on the Internet

    [MIT: Introduction to Computer Science and Programming in Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/video_galleries/lecture-videos/) — start with computation, branching, and iteration; the videos explain how to reason about code before running it.

    ## References

    - [Python tutorial: errors and exceptions](https://docs.python.org/3/tutorial/errors.html) — distinguish expected input failures from bugs. Follow with [Python tutorial](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/python_tutorial.py) for arrays and files.
    """)
    return


if __name__ == "__main__":
    app.run()
