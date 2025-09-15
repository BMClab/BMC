import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Text and binary files

        There are two kinds of computer files: text and binary files. Text files are structured as a sequence of lines of electronic text. The most common formats of a text file are ASCII (with 128 ($2^7$)) different characters and UTF-8 (which includes non-English characters). A binary file is a file that is not structured as a text file. Because in fact everything in a computer is stored in binary format (a sequence of zeros and ones), text files are binary files that store text codes.

        To open and read a text file is simple and straightforward. A text file doesn't need additional information to be read, and can be opened by any text-processing software. This is not the case of a binary file, we need to have extra information about how the file is structured to be able to read it. However, binary files can store more information per file size than text files and we can read and write binary files faster than text files. This is one of the reasons why software developers would choose a binary format.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
