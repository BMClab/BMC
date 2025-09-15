import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code structure for data analysis

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
        Sometimes data from experiments are stored in different files where each file contains data for different subjects, trials, conditions, etc. This text presents a common and simple solution to write a code to analyze such data.  
        The basic idea is that the name of the file is created in a structured way and you can use that to run a sequence of procedures inside one or more nested loops.   
        For instance, consider that the two first letters of the filename encode the initials of the subject's name, the next two letters the different conditions, and the last two characters the trial number. 
        """
    )
    return


@app.cell
def _():
    subjects   = ['AA', 'AB']
    conditions = ['c1', 'c2']
    trials     = ['01', '02', '03']
    return conditions, subjects, trials


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We could open and process these files with:
        """
    )
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
    mo.md(
        r"""
        The problem with this code is that if one one more files are missing or corrupted (which is typical), it will break. A solution is to read the file inside a `try` function. The `try...except` handles exceptions such as a failure in reading a file and then we can use a `continue` statement to skip each failed iteration in the inner loop.  
        Let's create some files and implement this idea.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Read and save files

        If the data is in text ([ASCII](http://en.wikipedia.org/wiki/ASCII)) format, it's easier to read the file with the [`Numpy`](http://www.numpy.org/) function [`loadtxt`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html) or with the [`pandas`](http://pandas.pydata.org/) function [`read_csv`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.io.parsers.read_csv.html). Both functions behave similarly; they can skip a certain number of first rows, can read files with different column separators, read numbers and letters, etc. `read_csv` tends to be faster but it returns a `pandas` [`DataFrame`](http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.html) object, which might not be useful if you are not into `pandas` (but you should be).

        To save data to a file, we can use the counterpart functions `savetxt` and `to_csv`:
        """
    )
    return


@app.cell
def _(conditions, subjects, trials):
    import numpy as np
    path = './../data/'
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
    mo.md(
        r"""
        In my case I used the './../' command to move up one directory relatively to my current directory (see the <a href="http://en.wikipedia.org/wiki/Cd_(command)">cd (command)</a>).  
        Let's remove one of the files:
        """
    )
    return


@app.cell
def _():
    import os
    os.remove('./../data/AAc202.txt')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's read the data in these files and handle a possible missing or corrupted file:
        """
    )
    return


@app.cell
def _(conditions, extension, np, path, subjects, trials):
    for _subject in subjects:
        for _condition in conditions:
            for _trial in trials:
                _filename = path + _subject + _condition + _trial + extension
                try:
                    _data = np.loadtxt(_filename, skiprows=1)
                except Exception as err:
                    print(_filename, err)
                    continue
                else:
                    print(_filename, 'loaded')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Store results

        The results of the analysis for each file can be stored in a variable in different ways.  
        We can store the results in a multidimensional variable where each dimension corresponds to the different indices in the loops. With the data above this would produce `results(s, c, t)`, a 2x2x3 array. Or we can store everything in a two-dimensional array where for example each row corresponds to each combination of subject, condition, and trial.   
        Let's try both ways:
        """
    )
    return


@app.cell
def _(conditions, extension, np, path, subjects, trials):
    results = np.empty(shape=(2, 2, 3, 3)) * np.NaN
    for (_s, _subject) in enumerate(subjects):
        for (_c, _condition) in enumerate(conditions):
            for (_t, _trial) in enumerate(trials):
                _filename = path + _subject + _condition + _trial + extension
                try:
                    _data = np.loadtxt(_filename, skiprows=1)
                except Exception as err:
                    continue
                else:
                    pass
                results[_s, _c, _t, :] = np.mean(_data, axis=0)
    print(results.shape)
    print(results)
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One problem with this approach is that for many dimensions the data gets convoluted and it might be difficult to read it.  
        The results for the first subject, condition, and trial are:
        """
    )
    return


@app.cell
def _(results):
    results[0, 0, 0, :]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can use the second approach and store the results in a two-dimensional array:
        """
    )
    return


@app.cell
def _(conditions, extension, np, path, subjects, trials):
    results_1 = np.empty(shape=(2 * 2 * 3, 3)) * np.NaN
    results2 = np.empty(shape=(2 * 2 * 3, 3)) * np.NaN
    _ind = 0
    for (_s, _subject) in enumerate(subjects):
        for (_c, _condition) in enumerate(conditions):
            for (_t, _trial) in enumerate(trials):
                _ind = _ind + 1
                _filename = path + _subject + _condition + _trial + extension
                try:
                    _data = np.loadtxt(_filename, skiprows=1)
                except Exception as err:
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
    mo.md(
        r"""
        We can create columns identifying the subject, condition, and trial, which might be useful for running statistical analysis:
        """
    )
    return


@app.cell
def _(conditions, extension, np, path, subjects, trials):
    results_2 = np.empty(shape=(2 * 2 * 3, 3)) * np.NaN
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
                except Exception as err:
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
    mo.md(
        r"""
        These are just some possible generic approaches to analyze data in multiple files.

        And we can save the results in a file:
        """
    )
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


if __name__ == "__main__":
    app.run()
