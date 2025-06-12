import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Tutorial on Python for scientific computing

        > Marcos Duarte, Renato Naville Watanabe  
        > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/pt/)  
        > Federal University of ABC, Brazil

        <p style="text-align: right;">A <a href="https://jupyter.org/">Jupyter Notebook</a></p>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Scope-of-this-tutorial" data-toc-modified-id="Scope-of-this-tutorial-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Scope of this tutorial</a></span></li><li><span><a href="#Python-as-a-calculator" data-toc-modified-id="Python-as-a-calculator-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Python as a calculator</a></span></li><li><span><a href="#The-import-function" data-toc-modified-id="The-import-function-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>The import function</a></span></li><li><span><a href="#Object-oriented-programming" data-toc-modified-id="Object-oriented-programming-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Object-oriented programming</a></span></li><li><span><a href="#Python-and-IPython-help" data-toc-modified-id="Python-and-IPython-help-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Python and IPython help</a></span><ul class="toc-item"><li><span><a href="#Tab-completion-in-IPython" data-toc-modified-id="Tab-completion-in-IPython-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Tab completion in IPython</a></span></li><li><span><a href="#The-four-most-helpful-commands-in-IPython" data-toc-modified-id="The-four-most-helpful-commands-in-IPython-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>The four most helpful commands in IPython</a></span></li><li><span><a href="#Comments" data-toc-modified-id="Comments-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Comments</a></span></li><li><span><a href="#Magic-functions" data-toc-modified-id="Magic-functions-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Magic functions</a></span></li></ul></li><li><span><a href="#Assignment-and-expressions" data-toc-modified-id="Assignment-and-expressions-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Assignment and expressions</a></span></li><li><span><a href="#Variables-and-types" data-toc-modified-id="Variables-and-types-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Variables and types</a></span><ul class="toc-item"><li><span><a href="#Numbers:-int,-float,-complex" data-toc-modified-id="Numbers:-int,-float,-complex-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Numbers: int, float, complex</a></span></li><li><span><a href="#Strings" data-toc-modified-id="Strings-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Strings</a></span></li><li><span><a href="#len()" data-toc-modified-id="len()-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>len()</a></span></li><li><span><a href="#Lists" data-toc-modified-id="Lists-7.4"><span class="toc-item-num">7.4&nbsp;&nbsp;</span>Lists</a></span></li><li><span><a href="#Tuples" data-toc-modified-id="Tuples-7.5"><span class="toc-item-num">7.5&nbsp;&nbsp;</span>Tuples</a></span></li><li><span><a href="#Sets" data-toc-modified-id="Sets-7.6"><span class="toc-item-num">7.6&nbsp;&nbsp;</span>Sets</a></span></li><li><span><a href="#Dictionaries" data-toc-modified-id="Dictionaries-7.7"><span class="toc-item-num">7.7&nbsp;&nbsp;</span>Dictionaries</a></span></li></ul></li><li><span><a href="#Built-in-Constants" data-toc-modified-id="Built-in-Constants-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Built-in Constants</a></span></li><li><span><a href="#Logical-(Boolean)-operators" data-toc-modified-id="Logical-(Boolean)-operators-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Logical (Boolean) operators</a></span><ul class="toc-item"><li><span><a href="#and,-or,-not" data-toc-modified-id="and,-or,-not-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>and, or, not</a></span></li><li><span><a href="#Comparisons" data-toc-modified-id="Comparisons-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>Comparisons</a></span></li></ul></li><li><span><a href="#Indentation-and-whitespace" data-toc-modified-id="Indentation-and-whitespace-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Indentation and whitespace</a></span></li><li><span><a href="#Control-of-flow" data-toc-modified-id="Control-of-flow-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Control of flow</a></span><ul class="toc-item"><li><span><a href="#if...elif...else" data-toc-modified-id="if...elif...else-11.1"><span class="toc-item-num">11.1&nbsp;&nbsp;</span><code>if</code>...<code>elif</code>...<code>else</code></a></span></li><li><span><a href="#for" data-toc-modified-id="for-11.2"><span class="toc-item-num">11.2&nbsp;&nbsp;</span>for</a></span><ul class="toc-item"><li><span><a href="#The-range()-function" data-toc-modified-id="The-range()-function-11.2.1"><span class="toc-item-num">11.2.1&nbsp;&nbsp;</span>The <code>range()</code> function</a></span></li></ul></li><li><span><a href="#while" data-toc-modified-id="while-11.3"><span class="toc-item-num">11.3&nbsp;&nbsp;</span>while</a></span></li></ul></li><li><span><a href="#Function-definition" data-toc-modified-id="Function-definition-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Function definition</a></span></li><li><span><a href="#Numeric-data-manipulation-with-Numpy" data-toc-modified-id="Numeric-data-manipulation-with-Numpy-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Numeric data manipulation with Numpy</a></span><ul class="toc-item"><li><span><a href="#Interpolation" data-toc-modified-id="Interpolation-13.1"><span class="toc-item-num">13.1&nbsp;&nbsp;</span>Interpolation</a></span></li></ul></li><li><span><a href="#Read-and-save-files" data-toc-modified-id="Read-and-save-files-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Read and save files</a></span></li><li><span><a href="#Ploting-with-matplotlib" data-toc-modified-id="Ploting-with-matplotlib-15"><span class="toc-item-num">15&nbsp;&nbsp;</span>Ploting with matplotlib</a></span></li><li><span><a href="#Signal-processing-with-Scipy" data-toc-modified-id="Signal-processing-with-Scipy-16"><span class="toc-item-num">16&nbsp;&nbsp;</span>Signal processing with Scipy</a></span></li><li><span><a href="#Symbolic-mathematics-with-Sympy" data-toc-modified-id="Symbolic-mathematics-with-Sympy-17"><span class="toc-item-num">17&nbsp;&nbsp;</span>Symbolic mathematics with Sympy</a></span></li><li><span><a href="#Data-analysis-with-pandas" data-toc-modified-id="Data-analysis-with-pandas-18"><span class="toc-item-num">18&nbsp;&nbsp;</span>Data analysis with pandas</a></span></li><li><span><a href="#More-about-Python" data-toc-modified-id="More-about-Python-19"><span class="toc-item-num">19&nbsp;&nbsp;</span>More about Python</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Scope of this tutorial

        This will be a very brief tutorial on Python.  
        For a more complete tutorial about Python see [A Whirlwind Tour of Python](https://github.com/jakevdp/WhirlwindTourOfPython) and [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) for a specific tutorial about Python for scientific computing.

        To use Python for scientific computing we need the Python program itself with its main modules and specific packages for scientific computing. [See this notebook on how to install Python for scientific computing](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/PythonInstallation.ipynb).  
        Once you get Python and the necessary packages for scientific computing ready to work, there are different ways to run Python, the main ones are:

        - open a terminal window in your computer and type `python` or `ipython` that the Python interpreter will start
        - run the `Jupyter notebook` and start working with Python in a browser
        - run `Spyder`, an interactive development environment (IDE)
        - run the `Jupyter qtconsole`, a more featured terminal
        - run Python online in a website such as [https://www.pythonanywhere.com/](https://www.pythonanywhere.com/) or [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)
        - run Python using any other Python editor or IDE

        We will use the Jupyter Notebook for this tutorial but you can run almost all the things we will see here using the other forms listed above.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Python as a calculator

        Once in the Jupyter notebook, if you type a simple mathematical expression and press `Shift+Enter` it will give the result of the expression:
        """
    )
    return


@app.cell
def _():
    1 + 2 - 25
    return


@app.cell
def _():
    4/7
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Using the `print` function, let's explore the mathematical operations available in Python:
        """
    )
    return


@app.cell
def _():
    print('1+2 = ', 1+2, '\n', '4*5 = ', 4*5, '\n', '6/7 = ', 6/7, '\n', '8**2 = ', 8**2, sep='')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And if we want the square-root of a number:
        """
    )
    return


@app.cell
def _(sqrt):
    sqrt(9)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We get an error message saying that the `sqrt` function if not defined. This is because `sqrt` and other mathematical functions are available with the `math` module:
        """
    )
    return


@app.cell
def _():
    import math
    return (math,)


@app.cell
def _(math):
    math.sqrt(9)
    return


@app.cell
def _():
    from math import sqrt
    return (sqrt,)


@app.cell
def _(sqrt):
    sqrt(9)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The import function

        We used the command '`import`' to be able to call certain functions. In Python functions are organized in modules and packages and they have to be imported in order to be used.   

        A module is a file containing Python definitions (e.g., functions) and statements. Packages are a way of structuring Python’s module namespace by using “dotted module names”. For example, the module name A.B designates a submodule named B in a package named A. To be used, modules and packages have to be imported in Python with the import function.   

        Namespace is a container for a set of identifiers (names), and allows the disambiguation of homonym identifiers residing in different namespaces. For example, with the command import math, we will have all the functions and statements defined in this module in the namespace '`math.`', for example, '`math.pi`' is the π constant and '`math.cos()`', the cosine function.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        By the way, to know which Python version you are running, we can use one of the following modules:
        """
    )
    return


@app.cell
def _():
    import sys
    sys.version
    return (sys,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And if you are in an IPython session:
        """
    )
    return


@app.cell
def _():
    from IPython import sys_info
    print(sys_info())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The first option gives information about the Python version; the latter also includes the IPython version, operating system, etc.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Object-oriented programming

        Python is designed as an object-oriented programming (OOP) language. OOP is a paradigm that represents concepts as "objects" that have data fields (attributes that describe the object) and associated procedures known as methods.

        This means that all elements in Python are objects and they have attributes which can be acessed with the dot (.) operator after the name of the object. We already experimented with that when we imported the module `sys`, it became an object, and we acessed one of its attribute: `sys.version`.

        OOP as a paradigm is much more than defining objects, attributes, and methods, but for now this is enough to get going with Python.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Python and IPython help

        To get help about any Python command, use `help()`:
        """
    )
    return


@app.cell
def _(math):
    help(math.degrees)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Or if you are in the IPython environment, simply add '?' to the function that a window will open at the bottom of your browser with the same help content:
        """
    )
    return


app._unparsable_cell(
    r"""
    math.degrees?
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And if you add a second '?' to the statement you get access to the original script file of the function (an advantage of an open source language), unless that function is a built-in function that does not have a script file, which is the case of the standard modules in Python (but you can access the Python source code if you want; it just does not come with the standard program for installation).

        So, let's see this feature with another function:
        """
    )
    return


app._unparsable_cell(
    r"""
    import scipy.fftpack
    scipy.fftpack.fft??
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To know all the attributes of an object, for example all the functions available in `math`, we can use the function `dir`:
        """
    )
    return


@app.cell
def _(math):
    print(dir(math))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Tab completion in IPython

        IPython has tab completion: start typing the name of the command (object) and press `tab` to see the names of objects available with these initials letters. When the name of the object is typed followed by a dot (`math.`), pressing `tab` will show all available attribites, scroll down to the desired attribute and press `Enter` to select it.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The four most helpful commands in IPython

        These are the most helpful commands in IPython (from [IPython tutorial](http://ipython.org/ipython-doc/dev/interactive/tutorial.html)):

         - `?` : Introduction and overview of IPython’s features.
         - `%quickref` : Quick reference.
         - `help` : Python’s own help system.
         - `object?` : Details about ‘object’, use ‘object??’ for extra details.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Comments

        Comments in Python start with the hash character, #, and extend to the end of the physical line:
        """
    )
    return


@app.cell
def _(math):
    # Import the math library to access more math stuff
    math.pi  # this is the pi constant; a useless comment since this is obvious
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To insert comments spanning more than one line, use a multi-line string with a pair of matching triple-quotes: `\"\"\"` or `'''` (we will see the string data type later). A typical use of a multi-line comment is as documentation strings and are meant for anyone reading the code:
        """
    )
    return


@app.cell
def _():
    """Documentation strings are typically written like that.

    A docstring is a string literal that occurs as the first statement
    in a module, function, class, or method definition.

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A docstring like above is useless and its output as a standalone statement looks uggly in IPython Notebook, but you will see its real importance when reading and writting codes.

        Commenting a programming code is an important step to make the code more readable, which Python cares a lot.   
        There is a style guide for writting Python code ([PEP 8](https://www.python.org/dev/peps/pep-0008/)) with a session about [how to write comments](https://www.python.org/dev/peps/pep-0008/#comments).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Magic functions

        IPython has a set of predefined ‘magic functions’ that you can call with a command line style syntax.   
        There are two kinds of magics, line-oriented and cell-oriented.   
        Line magics are prefixed with the % character and work much like OS command-line calls: they get as an argument the rest of the line, where arguments are passed without parentheses or quotes.   
        Cell magics are prefixed with a double %%, and they are functions that get as an argument not only the rest of the line, but also the lines below it in a separate argument.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Assignment and expressions

        The equal sign ('=') is used to assign a value to a variable. Afterwards, no result is displayed before the next interactive prompt:
        """
    )
    return


@app.cell
def _():
    x = 1
    return (x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Spaces between the statements are optional but it helps for readability.

        To see the value of the variable, call it again or use the print function:
        """
    )
    return


@app.cell
def _(x):
    x
    return


@app.cell
def _(x):
    print(x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Of course, the last assignment is that holds:
        """
    )
    return


@app.cell
def _():
    x_1 = 2
    x_1 = 3
    x_1
    return (x_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In mathematics '=' is the symbol for identity, but in computer programming '=' is used for assignment, it means that the right part of the expresssion is assigned to its left part.   
        For example, 'x=x+1' does not make sense in mathematics but it does in computer programming:
        """
    )
    return


@app.cell
def _(x_1):
    x_2 = x_1 + 1
    print(x_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A value can be assigned to several variables simultaneously:
        """
    )
    return


@app.cell
def _():
    x_3 = y = 4
    print(x_3)
    print(y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Several values can be assigned to several variables at once:
        """
    )
    return


@app.cell
def _():
    (x_4, y_1) = (5, 6)
    print(x_4)
    print(y_1)
    return x_4, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And with that, you can do (!):
        """
    )
    return


@app.cell
def _(x_4, y_1):
    (x_5, y_2) = (y_1, x_4)
    print(x_5)
    print(y_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Variables must be “defined” (assigned a value) before they can be used, or an error will occur:
        """
    )
    return


@app.cell
def _(z):
    x_6 = z
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Variables and types

        There are different types of built-in objects in Python (and remember that everything in Python is an object):
        """
    )
    return


@app.cell
def _():
    import types
    print(dir(types))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's see some of them now.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Numbers: int, float, complex

        Numbers can an integer (int), float, and complex (with imaginary part).   
        Let's use the function `type` to show the type of number (and later for any other object):
        """
    )
    return


@app.cell
def _():
    type(6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A float is a non-integer number:
        """
    )
    return


@app.cell
def _(math):
    math.pi
    return


@app.cell
def _(math):
    type(math.pi)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Python (IPython) is showing `math.pi` with only 15 decimal cases, but internally a float is represented with higher precision.   
        Floating point numbers in Python are implemented using a double (eight bytes) word; the precison and internal representation of floating point numbers are machine specific and are available in:
        """
    )
    return


@app.cell
def _(sys):
    sys.float_info
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Be aware that floating-point numbers can be trick in computers:
        """
    )
    return


@app.cell
def _():
    0.1 + 0.2
    return


@app.cell
def _():
    0.1 + 0.2 - 0.3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        These results are not correct (and the problem is not due to Python). The error arises from the fact that floating-point numbers are represented in computer hardware as base 2 (binary) fractions and most decimal fractions cannot be represented exactly as binary fractions. As consequence, decimal floating-point numbers are only approximated by the binary floating-point numbers actually stored in the machine. [See here for more on this issue](http://docs.python.org/2/tutorial/floatingpoint.html).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A complex number has real and imaginary parts:
        """
    )
    return


@app.cell
def _():
    1 + 2j
    return


@app.cell
def _():
    print(type(1+2j))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Each part of a complex number is represented as a floating-point number. We can see them using the attributes `.real` and `.imag`:
        """
    )
    return


@app.cell
def _():
    print((1 + 2j).real)
    print((1 + 2j).imag)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Strings

        Strings can be enclosed in single quotes or double quotes:
        """
    )
    return


@app.cell
def _():
    s = 'string (str) is a built-in type in Python'
    s
    return (s,)


@app.cell
def _(s):
    type(s)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        String enclosed with single and double quotes are equal, but it may be easier to use one instead of the other:
        """
    )
    return


app._unparsable_cell(
    r"""
    'string (str) is a Python's built-in type'
    """,
    name="_"
)


@app.cell
def _():
    "string (str) is a Python's built-in type"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        But you could have done that using the Python escape character '\':
        """
    )
    return


@app.cell
def _():
    'string (str) is a Python\'s built-in type'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Strings can be concatenated (glued together) with the + operator, and repeated with *:
        """
    )
    return


@app.cell
def _():
    s_1 = 'P' + 'y' + 't' + 'h' + 'o' + 'n'
    print(s_1)
    print(s_1 * 5)
    return (s_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Strings can be subscripted (indexed); like in C, the first character of a string has subscript (index) 0:
        """
    )
    return


@app.cell
def _(s_1):
    print('s[0] = ', s_1[0], '  (s[index], start at 0)')
    print('s[5] = ', s_1[5])
    print('s[-1] = ', s_1[-1], '  (last element)')
    print('s[:] = ', s_1[:], '  (all elements)')
    print('s[1:] = ', s_1[1:], '  (from this index (inclusive) till the last (inclusive))')
    print('s[2:4] = ', s_1[2:4], '  (from first index (inclusive) till second index (exclusive))')
    print('s[:2] = ', s_1[:2], '  (till this index, exclusive)')
    print('s[:10] = ', s_1[:10], '  (Python handles the index if it is larger than the string length)')
    print('s[-10:] = ', s_1[-10:])
    print('s[0:5:2] = ', s_1[0:5:2], '  (s[ini:end:step])')
    print('s[::2] = ', s_1[::2], '  (s[::step], initial and final indexes can be omitted)')
    print('s[0:5:-1] = ', s_1[::-1], '  (s[::-step] reverses the string)')
    print('s[:2] + s[2:] = ', s_1[:2] + s_1[2:], '  (because of Python indexing, this sounds natural)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### len()

        Python has a built-in functon to get the number of itens of a sequence:
        """
    )
    return


@app.cell
def _():
    help(len)
    return


@app.cell
def _():
    s_2 = 'Python'
    len(s_2)
    return (s_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The function len() helps to understand how the backward indexing works in Python.    
        The index s[-i] should be understood as s[len(s) - i] rather than accessing directly the i-th element from back to front. This is why the last element of a string is s[-1]:
        """
    )
    return


@app.cell
def _(s_2):
    print('s = ', s_2)
    print('len(s) = ', len(s_2))
    print('len(s)-1 = ', len(s_2) - 1)
    print('s[-1] = ', s_2[-1])
    print('s[len(s) - 1] = ', s_2[len(s_2) - 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Or, strings can be surrounded in a pair of matching triple-quotes: \"\"\" or '''. End of lines do not need to be escaped when using triple-quotes, but they will be included in the string. This is how we created a multi-line comment earlier:
        """
    )
    return


@app.cell
def _():
    """Strings can be surrounded in a pair of matching triple-quotes: \""" or '''.

    End of lines do not need to be escaped when using triple-quotes,
    but they will be included in the string.

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Lists

        Values can be grouped together using different types, one of them is list, which can be written as a list of comma-separated values between square brackets. List items need not all have the same type:
        """
    )
    return


@app.cell
def _():
    x_7 = ['spam', 'eggs', 100, 1234]
    x_7
    return (x_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Lists can be indexed and the same indexing rules we saw for strings are applied:
        """
    )
    return


@app.cell
def _(x_7):
    x_7[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The function len() works for lists:
        """
    )
    return


@app.cell
def _(x_7):
    len(x_7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Tuples

        A tuple consists of a number of values separated by commas, for instance:
        """
    )
    return


@app.cell
def _():
    t = ('spam', 'eggs', 100, 1234)
    t
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The type tuple is why multiple assignments in a single line works; elements separated by commas (with or without surrounding parentheses) are a tuple and in an expression with an '=', the right-side tuple is attributed to the left-side tuple:
        """
    )
    return


@app.cell
def _():
    a, b = 1, 2
    print('a = ', a, '\nb = ', b)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Is the same as:
        """
    )
    return


@app.cell
def _():
    (a_1, b_1) = (1, 2)
    print('a = ', a_1, '\nb = ', b_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sets

        Python also includes a data type for sets. A set is an unordered collection with no duplicate elements.
        """
    )
    return


@app.cell
def _():
    basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
    fruit = set(basket)  # create a set without duplicates
    fruit
    return (fruit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As set is an unordered collection, it can not be indexed as lists and tuples.
        """
    )
    return


@app.cell
def _(fruit):
    set(['orange', 'pear', 'apple', 'banana'])
    'orange' in fruit  # fast membership testing
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Dictionaries

        Dictionary is a collection of elements organized keys and values. Unlike lists and tuples, which are indexed by a range of numbers, dictionaries are indexed by their keys:
        """
    )
    return


@app.cell
def _():
    tel = {'jack': 4098, 'sape': 4139}
    tel
    return (tel,)


@app.cell
def _(tel):
    tel['guido'] = 4127
    tel
    return


@app.cell
def _(tel):
    tel['jack']
    return


@app.cell
def _(tel):
    del tel['sape']
    tel['irv'] = 4127
    tel
    return


@app.cell
def _(tel):
    tel.keys()
    return


@app.cell
def _(tel):
    'guido' in tel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The dict() constructor builds dictionaries directly from sequences of key-value pairs:
        """
    )
    return


@app.cell
def _():
    tel_1 = dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
    tel_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Built-in Constants

        - **False** : false value of the bool type
        - **True** : true value of the bool type
        - **None** : sole value of types.NoneType. None is frequently used to represent the absence of a value.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In computer science, the Boolean or logical data type is composed by two values, true and false, intended to represent the values of logic and Boolean algebra. In Python, 1 and 0 can also be used in most situations as equivalent to the Boolean values.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Logical (Boolean) operators
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### and, or, not
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - **and** : logical AND operator. If both the operands are true then condition becomes true.	 (a and b) is true.
        - **or** : logical OR Operator. If any of the two operands are non zero then condition becomes true.	 (a or b) is true.
        - **not** : logical NOT Operator. Reverses the logical state of its operand. If a condition is true then logical NOT operator will make false.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Comparisons

        The following comparison operations are supported by objects in Python:

        - **==** : equal
        - **!=** : not equal
        - **<**	: strictly less than
        - **<=** : less than or equal
        - **\>** : strictly greater than
        - **\>=** : greater than or equal
        - **is** : object identity
        - **is not** : negated object identity
        """
    )
    return


@app.cell
def _():
    True == False
    return


@app.cell
def _():
    not True == False
    return


@app.cell
def _():
    1 < 2 > 1
    return


@app.cell
def _():
    True != (False or True)
    return


@app.cell
def _():
    True != False or True
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Indentation and whitespace

        In Python, statement grouping is done by indentation (this is mandatory), which are done by inserting whitespaces, not tabs. Indentation is also recommended for alignment of function calling that span more than one line for better clarity.   
        We will see examples of indentation in the next session.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Control of flow

        ### `if`...`elif`...`else`

        Conditional statements (to peform something if another thing is True or False) can be implemmented using the `if` statement:
        ```
        if expression:
           statement
        elif:
           statement     
        else:
           statement
        ```
        `elif` (one or more) and `else` are optionals.   
        The indentation is obligatory.   
        For example:
        """
    )
    return


@app.cell
def _():
    if True:
        pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Which does nothing useful.   

        Let's use the `if`...`elif`...`else` statements to categorize the [body mass index](http://en.wikipedia.org/wiki/Body_mass_index) of a person:
        """
    )
    return


@app.cell
def _():
    # body mass index
    weight = 100  # kg
    height = 1.70  # m
    bmi = weight / height**2
    return bmi, height, weight


@app.cell
def _(bmi, height, weight):
    if bmi < 15:
        c = 'very severely underweight'
    elif 15 <= bmi < 16:
        c = 'severely underweight'
    elif 16 <= bmi < 18.5:
        c = 'underweight'
    elif 18.5 <= bmi < 25:
        c = 'normal'
    elif 25 <= bmi < 30:
        c = 'overweight'
    elif 30 <= bmi < 35:
        c = 'moderately obese'
    elif 35 <= bmi < 40:
        c = 'severely obese'
    else:
        c = 'very severely obese'

    print('For a weight of {0:.1f} kg and a height of {1:.2f} m,\n\
    the body mass index (bmi) is {2:.1f} kg/m2,\nwhich is considered {3:s}.'\
          .format(weight, height, bmi, c))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### for

        The `for` statement iterates over a sequence to perform operations (a loop event).
        ```
        for iterating_var in sequence:
            statements
        ```
        """
    )
    return


@app.cell
def _():
    for i in [3, 2, 1, 'go!']:
        print(i, end=', ')
    return


@app.cell
def _():
    for letter in 'Python':
        print(letter),
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### The `range()` function

        The built-in function range() is useful if we need to create a sequence of numbers, for example, to iterate over this list. It generates lists containing arithmetic progressions:
        """
    )
    return


@app.cell
def _():
    help(range)
    return


@app.cell
def _():
    range(10)
    return


@app.cell
def _():
    range(1, 10, 2)
    return


@app.cell
def _():
    for i_1 in range(10):
        n2 = i_1 ** 2
        (print(n2),)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### while

        The `while` statement is used for repeating sections of code in a loop until a condition is met (this different than the `for` statement which executes n times):
        ```
        while expression:
            statement
        ```
        Let's generate the Fibonacci series using a `while` loop:
        """
    )
    return


@app.cell
def _():
    (a_2, b_2) = (0, 1)
    while b_2 < 1000:
        print(b_2, end='   ')
        (a_2, b_2) = (b_2, a_2 + b_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Function definition

        A function in a programming language is a piece of code that performs a specific task. Functions are used to reduce duplication of code making easier to reuse it and to decompose complex problems into simpler parts. The use of functions contribute to the clarity of the code.

        A function is created with the `def` keyword and the statements in the block of the function must be indented:
        """
    )
    return


@app.function
def function():
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As per construction, this function does nothing when called:
        """
    )
    return


@app.cell
def _():
    function()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The general syntax of a function definition is:
        ```
        def function_name( parameters ):
           \"\"\"Function docstring.

           The help for the function

           \"\"\"

           function body

           return variables
        ```
        A more useful function:
        """
    )
    return


@app.function
def fibo(N):
    """Fibonacci series: the sum of two elements defines the next.

    The series is calculated till the input parameter N and
    returned as an ouput variable.

    """

    a, b, c = 0, 1, []
    while b < N:
        c.append(b)
        a, b = b, a + b

    return c


@app.cell
def _():
    fibo(100)
    return


@app.cell
def _():
    if 3 > 2:
           print('teste')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's implemment the body mass index calculus and categorization as a function:
        """
    )
    return


@app.function
def bmi_1(weight, height):
    """Body mass index calculus and categorization.

    Enter the weight in kg and the height in m.
    See http://en.wikipedia.org/wiki/Body_mass_index

    """
    bmi = weight / height ** 2
    if bmi < 15:
        c = 'very severely underweight'
    elif 15 <= bmi < 16:
        c = 'severely underweight'
    elif 16 <= bmi < 18.5:
        c = 'underweight'
    elif 18.5 <= bmi < 25:
        c = 'normal'
    elif 25 <= bmi < 30:
        c = 'overweight'
    elif 30 <= bmi < 35:
        c = 'moderately obese'
    elif 35 <= bmi < 40:
        c = 'severely obese'
    else:
        c = 'very severely obese'
    s = 'For a weight of {0:.1f} kg and a height of {1:.2f} m,    the body mass index (bmi) is {2:.1f} kg/m2,    which is considered {3:s}.'.format(weight, height, bmi, c)
    print(s)


@app.cell
def _():
    bmi_1(73, 1.7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Numeric data manipulation with Numpy

        Numpy is the fundamental package for scientific computing in Python and has a N-dimensional array package convenient to work with numerical data. With Numpy it's much easier and faster to work with numbers grouped as 1-D arrays (a vector), 2-D arrays (like a table or matrix), or higher dimensions. Let's create 1-D and 2-D arrays in Numpy:
        """
    )
    return


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(np):
    x1d = np.array([1, 2, 3, 4, 5, 6])
    print(type(x1d))
    x1d
    return (x1d,)


@app.cell
def _(np):
    x2d = np.array([[1, 2, 3], [4, 5, 6]])
    x2d
    return (x2d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        len() and the Numpy functions size() and shape() give information aboout the number of elements and the structure of the Numpy array:
        """
    )
    return


@app.cell
def _(np, x1d, x2d):
    print('1-d array:')
    print(x1d)
    print('len(x1d) = ', len(x1d))
    print('np.size(x1d) = ', np.size(x1d))
    print('np.shape(x1d) = ', np.shape(x1d))
    print('np.ndim(x1d) = ', np.ndim(x1d))
    print('\n2-d array:')
    print(x2d)
    print('len(x2d) = ', len(x2d))
    print('np.size(x2d) = ', np.size(x2d))
    print('np.shape(x2d) = ', np.shape(x2d))
    print('np.ndim(x2d) = ', np.ndim(x2d))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Create random data
        """
    )
    return


@app.cell
def _(np):
    x_8 = np.random.randn(4, 3)
    x_8
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Joining (stacking together) arrays
        """
    )
    return


@app.cell
def _(np):
    x_9 = np.random.randint(0, 5, size=(2, 3))
    print(x_9)
    y_3 = np.random.randint(5, 10, size=(2, 3))
    print(y_3)
    return x_9, y_3


@app.cell
def _(np, x_9, y_3):
    np.vstack((x_9, y_3))
    return


@app.cell
def _(np, x_9, y_3):
    np.hstack((x_9, y_3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Create equally spaced data
        """
    )
    return


@app.cell
def _(np):
    np.arange(start = 1, stop = 10, step = 2)
    return


@app.cell
def _(np):
    np.linspace(start = 0, stop = 1, num = 11)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Interpolation

        Consider the following data:
        """
    )
    return


@app.cell
def _():
    y_4 = [5, 4, 10, 8, 1, 10, 2, 7, 1, 3]
    return (y_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Suppose we want to create data in between the given data points (interpolation); for instance, let's try to double the resolution of the data by generating twice as many data:
        """
    )
    return


@app.cell
def _(np, y_4):
    t_1 = np.linspace(0, len(y_4), len(y_4))
    tn = np.linspace(0, len(y_4), 2 * len(y_4))
    yn = np.interp(tn, t_1, y_4)
    yn
    return t_1, tn, yn


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The key is the Numpy `interp` function, from its help:   

            interp(x, xp, fp, left=None, right=None)       
            One-dimensional linear interpolation.   
            Returns the one-dimensional piecewise linear interpolant to a function with given values at discrete data-points.

        A plot of the data will show what we have done:
        """
    )
    return


@app.cell
def _(t_1, tn, y_4, yn):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(t_1, y_4, 'bo-', lw=2, label='original data')
    plt.plot(tn, yn, '.-', color=[1, 0, 0, 0.5], lw=2, label='interpolated')
    plt.legend(loc='best', framealpha=0.5)
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For more about Numpy, see [http://www.numpy.org/](http://www.numpy.org/).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Read and save files

        There are two kinds of computer files: text files and binary files:
        > Text file: computer file where the content is structured as a sequence of lines of electronic text. Text files can contain plain text (letters, numbers, and symbols) but they are not limited to such. The type of content in the text file is defined by the Unicode encoding (a computing industry standard for the consistent encoding, representation and handling of text expressed in most of the world's writing systems).   
        >
        > Binary file: computer file where the content is encoded in binary form, a sequence of integers representing byte values.

        Let's see how to save and read numeric data stored in a text file:

        **Using plain Python**
        """
    )
    return


@app.cell
def _():
    f = open("newfile.txt", "w")           # open file for writing
    f.write("This is a test\n")            # save to file
    f.write("And here is another line\n")  # save to file
    f.close()
    f = open('newfile.txt', 'r')           # open file for reading
    f = f.read()                           # read from file
    print(f)
    return


@app.cell
def _():
    help(open)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Using Numpy**
        """
    )
    return


@app.cell
def _(np):
    data = np.random.randn(3,3)
    np.savetxt('myfile.txt', data, fmt="%12.6G")    # save to file
    data = np.genfromtxt('myfile.txt', unpack=True) # read from file
    data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Mount the Google Drive
        """
    )
    return


@app.cell
def _():
    from google.colab import drive
    drive.mount('/content/drive')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For example, using pandas to read a csv file in the Google Drive:
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    tabela = pd.read_csv('/content/sample_data/california_housing_test.csv')
    tabela
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Ploting with matplotlib

        Matplotlib is the most-widely used packge for plotting data in Python. Let's see some examples of it.
        """
    )
    return


@app.cell
def _(np, plt):
    t_2 = np.linspace(0, 0.99, 100)
    x_10 = np.sin(2 * np.pi * 2 * t_2)
    n = np.random.randn(100) / 5
    plt.Figure(figsize=(12, 8))
    plt.plot(t_2, x_10, label='sine', linewidth=2)
    plt.plot(t_2, x_10 + n, label='noisy sine', linewidth=2)
    plt.annotate(text='$sin(4 \\pi t)$', xy=(0.2, 1), fontsize=20, color=[0, 0, 1])
    plt.legend(loc='best', framealpha=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Data plotting using matplotlib')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Use the IPython magic `%matplotlib qt` to plot a figure in a separate window (from where you will be able to change some of the figure proprerties), **but it doesn't work in Google Colab**:
        """
    )
    return


@app.cell
def _():
    # '%matplotlib qt' command supported automatically in marimo
    return


@app.cell
def _(np, plt):
    (mu, sigma) = (10, 2)
    x_11 = mu + sigma * np.random.randn(1000)
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(x_11, 'ro')
    ax1.set_title('Data')
    ax1.grid()
    (n_1, bins, patches) = ax2.hist(x_11, 25, density=True, facecolor='r')
    ax2.set_xlabel('Bins')
    ax2.set_ylabel('Probability')
    ax2.set_title('Histogram')
    fig.suptitle('Another example using matplotlib', fontsize=18, y=1)
    ax2.grid()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And a window with the following figure should appear (**it doesn't work in Google Colab**):
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/plot.png?raw=1" alt="Plot"/></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You can switch back and forth between inline and separate figure using the `%matplotlib` magic commands used above. There are plenty more examples with the source code in the [matplotlib gallery](http://matplotlib.org/gallery.html).
        """
    )
    return


@app.cell
def _():
    # get back the inline plot
    # '%matplotlib inline' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Signal processing with Scipy

        The Scipy package has a lot of functions for signal processing, among them: Integration (scipy.integrate), Optimization (scipy.optimize), Interpolation (scipy.interpolate), Fourier Transforms (scipy.fftpack), Signal Processing (scipy.signal), Linear Algebra (scipy.linalg), and Statistics (scipy.stats). As an example, let's see how to use a low-pass Butterworth filter to attenuate high-frequency noise and how the differentiation process of a signal affects the signal-to-noise content. We will also calculate the Fourier transform of these data to look at their frequencies content.
        """
    )
    return


@app.cell
def _(np, scipy):
    from scipy.signal import butter, filtfilt
    freq = 100.0
    t_3 = np.arange(0, 1, 0.01)
    w = 2 * np.pi * 1
    y_5 = np.sin(w * t_3) + 0.1 * np.sin(10 * w * t_3)
    (b_3, a_3) = butter(4, 5 / (freq / 2), btype='low')
    y2 = filtfilt(b_3, a_3, y_5)
    ydd = np.diff(y_5, 2) * freq * freq
    y2dd = np.diff(y2, 2) * freq * freq
    yfft = np.abs(scipy.fftpack.fft(y_5)) / (y_5.size / 2)
    y2fft = np.abs(scipy.fftpack.fft(y2)) / (y_5.size / 2)
    freqs = scipy.fftpack.fftfreq(y_5.size, 1.0 / freq)
    yddfft = np.abs(scipy.fftpack.fft(ydd)) / (ydd.size / 2)
    y2ddfft = np.abs(scipy.fftpack.fft(y2dd)) / (ydd.size / 2)
    freqs2 = scipy.fftpack.fftfreq(ydd.size, 1.0 / freq)
    return freqs, t_3, y2, y2dd, y2ddfft, y2fft, y_5, ydd, yddfft, yfft


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the plots:
        """
    )
    return


@app.cell
def _(freqs, plt, t_3, y2, y2dd, y2ddfft, y2fft, y_5, ydd, yddfft, yfft):
    (fig_1, ((ax1_1, ax2_1), (ax3, ax4))) = plt.subplots(2, 2, figsize=(12, 6))
    ax1_1.set_title('Temporal domain', fontsize=14)
    ax1_1.plot(t_3, y_5, 'r', linewidth=2, label='raw data')
    ax1_1.plot(t_3, y2, 'b', linewidth=2, label='filtered @ 5 Hz')
    ax1_1.set_ylabel('f')
    ax1_1.legend(frameon=False, fontsize=12)
    ax2_1.set_title('Frequency domain', fontsize=14)
    ax2_1.plot(freqs[:int(yfft.size / 4)], yfft[:int(yfft.size / 4)], 'r', lw=2, label='raw data')
    ax2_1.plot(freqs[:int(yfft.size / 4)], y2fft[:int(yfft.size / 4)], 'b--', lw=2, label='filtered @ 5 Hz')
    ax2_1.set_ylabel('FFT(f)')
    ax2_1.legend(frameon=False, fontsize=12)
    ax3.plot(t_3[:-2], ydd, 'r', linewidth=2, label='raw')
    ax3.plot(t_3[:-2], y2dd, 'b', linewidth=2, label='filtered @ 5 Hz')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel("f ''")
    ax4.plot(freqs[:int(yddfft.size / 4)], yddfft[:int(yddfft.size / 4)], 'r', lw=2, label='raw')
    ax4.plot(freqs[:int(yddfft.size / 4)], y2ddfft[:int(yddfft.size / 4)], 'b--', lw=2, label='filtered @ 5 Hz')
    ax4.set_xlabel('Frequency [Hz]')
    ax4.set_ylabel("FFT(f '')")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For more about Scipy, see [https://docs.scipy.org/doc/scipy/reference/tutorial/](https://docs.scipy.org/doc/scipy/reference/tutorial/).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Symbolic mathematics with Sympy

        Sympy is a package to perform symbolic mathematics in Python. Let's see some of its features:
        """
    )
    return


@app.cell
def _():
    from IPython.display import display
    import sympy as sym
    from sympy.interactive import printing
    printing.init_printing()
    return display, sym


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Define some symbols and the create a second-order polynomial function (a.k.a., parabola):
        """
    )
    return


@app.cell
def _(sym):
    (x_12, y_6) = sym.symbols('x y')
    y_6 = x_12 ** 2 - 2 * x_12 - 3
    y_6
    return x_12, y_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Plot the parabola at some given range:
        """
    )
    return


@app.cell
def _(x_12, y_6):
    from sympy.plotting import plot
    plot(y_6, (x_12, -3, 5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the roots of the parabola are given by:
        """
    )
    return


@app.cell
def _(sym, x_12, y_6):
    sym.solve(y_6, x_12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can also do symbolic differentiation and integration:
        """
    )
    return


@app.cell
def _(sym, x_12, y_6):
    dy = sym.diff(y_6, x_12)
    dy
    return (dy,)


@app.cell
def _(dy, sym, x_12):
    sym.integrate(dy, x_12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For example, let's use Sympy to represent three-dimensional rotations. Consider the problem of a coordinate system xyz rotated in relation to other coordinate system XYZ. The single rotations around each axis are illustrated by:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/rotations.png?raw=1" alt="Rotations"/></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The single 3D rotation matrices around Z, Y, and X axes can be expressed in Sympy:
        """
    )
    return


@app.cell
def _(display):
    from IPython.core.display import Math
    from sympy import symbols, cos, sin, Matrix, latex
    (a_4, b_4, g) = symbols('alpha beta gamma')
    RX = Matrix([[1, 0, 0], [0, cos(a_4), -sin(a_4)], [0, sin(a_4), cos(a_4)]])
    display(Math('\\mathbf{R_{X}}=' + latex(RX, mat_str='matrix')))
    RY = Matrix([[cos(b_4), 0, sin(b_4)], [0, 1, 0], [-sin(b_4), 0, cos(b_4)]])
    display(Math('\\mathbf{R_{Y}}=' + latex(RY, mat_str='matrix')))
    RZ = Matrix([[cos(g), -sin(g), 0], [sin(g), cos(g), 0], [0, 0, 1]])
    display(Math('\\mathbf{R_{Z}}=' + latex(RZ, mat_str='matrix')))
    return Math, RX, RY, RZ, a_4, b_4, g, latex


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And using Sympy, a sequence of elementary rotations around X, Y, Z axes is given by:
        """
    )
    return


@app.cell
def _(Math, RX, RY, RZ, display, latex):
    RXYZ = RZ*RY*RX
    display(Math(r'\mathbf{R_{XYZ}}=' + latex(RXYZ, mat_str = 'matrix')))
    return (RXYZ,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Suppose there is a rotation only around X ($\alpha$) by$\pi/2$; we can get the numerical value of the rotation matrix by substituing the angle values:
        """
    )
    return


@app.cell
def _(RXYZ, a_4, b_4, g, np):
    r = RXYZ.subs({a_4: np.pi / 2, b_4: 0, g: 0})
    r
    return (r,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And we can prettify this result:
        """
    )
    return


@app.cell
def _(Math, display, latex, r):
    display(Math(r'\mathbf{R_{(\alpha=\pi/2)}}=' + latex(r.n(3, chop=True), mat_str = 'matrix')))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For more about Sympy, see [http://docs.sympy.org/latest/tutorial/](http://docs.sympy.org/latest/tutorial/).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Data analysis with pandas

        > "[pandas](http://pandas.pydata.org/) is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python."

        To work with labellled data, pandas has a type called DataFrame (basically, a matrix where columns and rows have may names and may be of different types) and it is also the main type of the software [R](http://www.r-project.org/). Fo ezample:
        """
    )
    return


@app.cell
def _():
    x_13 = 5 * ['A'] + 5 * ['B']
    x_13
    return


@app.cell
def _(np, pd):
    df = pd.DataFrame(np.random.rand(10, 2), columns=['Level 1', 'Level 2'])
    df['Group'] = pd.Series(['A'] * 5 + ['B'] * 5)
    plot_1 = df.boxplot(by='Group')
    return


@app.cell
def _(np, pd):
    from pandas.plotting import scatter_matrix
    df_1 = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
    plot_2 = scatter_matrix(df_1, alpha=0.5, figsize=(8, 6), diagonal='kde')
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        pandas is aware the data is structured and give you basic statistics considerint that and nicely formatted:
        """
    )
    return


@app.cell
def _(df_1):
    df_1.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For more on pandas, see this tutorial: [http://pandas.pydata.org/pandas-docs/stable/10min.html](http://pandas.pydata.org/pandas-docs/stable/10min.html).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Plot with some interactive features of matplotlib
        """
    )
    return


app._unparsable_cell(
    r"""
    !pip install -q ipympl
    from google.colab import output
    output.enable_custom_widget_manager()
    # '%matplotlib widget' command supported automatically in marimo
    """,
    name="_"
)


@app.cell
def _(np, plt):
    t_4 = np.linspace(0, 0.99, 100)
    x_14 = np.sin(2 * np.pi * 2 * t_4)
    n_2 = np.random.randn(100) / 5
    plt.Figure(figsize=(12, 8))
    plt.plot(t_4, x_14, label='sine', linewidth=2)
    plt.plot(t_4, x_14 + n_2, label='noisy sine', linewidth=2)
    plt.annotate(text='$sin(4 \\pi t)$', xy=(0.2, 1), fontsize=20, color=[0, 0, 1])
    plt.legend(loc='best', framealpha=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Data plotting using matplotlib')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interactive browser controls for notebooks
        """
    )
    return


app._unparsable_cell(
    r"""
    !pip install -q ipywidgets
    import ipywidgets
    """,
    name="_"
)


@app.cell
def _(ipywidgets, np, plt):
    def func(t, A, tau):
        """Saturating Exponential"""
        return A * (1 - np.exp(-t / tau))
    (fig_2, ax) = plt.subplots(figsize=(8, 4))
    ax.set_ylim([-0.1, 10])
    ax.grid(True)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    t_5 = np.linspace(0, 300, 101)
    V = func(t_5, 5, 50)

    @ipywidgets.interact(A=(0, 10, 0.5), tau=(1, 100, 1))
    def update(A=5, tau=50):
        """Remove old lines from plot and plot new one"""
        [l.remove() for l in ax.lines]
        ax.plot(t_5, func(t_5, A, tau), color='r', lw=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## More about Python

        There is a lot of good material in the internet about Python for scientific computing, here is a small list of interesting stuff:  

         - [How To Think Like A Computer Scientist](http://www.openbookproject.net/thinkcs/python/english2e/) or [the interactive edition](http://interactivepython.org/courselib/static/thinkcspy/index.html) (book)
         - [Python Scientific Lecture Notes](http://scipy-lectures.github.io/) (lecture notes)  
         - [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) (tutorial/book)    
         - [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki)
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
