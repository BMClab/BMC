import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Python for scientific computing

        > Marcos Duarte, Renato Naville Watanabe  
        > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)  
        > Federal University of ABC, Brazil

        <p style="text-align: right;">A <a href="https://jupyter.org/">Jupyter Notebook</a></p>
        """
    )
    return


@app.cell
def _():
    from IPython.display import Image
    Image(data='http://imgs.xkcd.com/comics/python.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Computing-as-a-third-kind-of-Science" data-toc-modified-id="Computing-as-a-third-kind-of-Science-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Computing as a third kind of Science</a></span></li><li><span><a href="#About-Python-[Python-documentation]" data-toc-modified-id="About-Python-[Python-documentation]-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>About Python [<a href="http://www.python.org/doc/essays/blurb/" rel="nofollow" target="_blank">Python documentation</a>]</a></span></li><li><span><a href="#About-Python-[Python-documentation]" data-toc-modified-id="About-Python-[Python-documentation]-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>About Python [<a href="http://www.python.org/doc/essays/blurb/" rel="nofollow" target="_blank">Python documentation</a>]</a></span></li><li><span><a href="#Glossary-for-the-Python-technical-characteristics-I" data-toc-modified-id="Glossary-for-the-Python-technical-characteristics-I-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Glossary for the Python technical characteristics I</a></span></li><li><span><a href="#Glossary-for-the-Python-technical-characteristics-II" data-toc-modified-id="Glossary-for-the-Python-technical-characteristics-II-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Glossary for the Python technical characteristics II</a></span></li><li><span><a href="#About-Python" data-toc-modified-id="About-Python-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>About Python</a></span></li><li><span><a href="#Python" data-toc-modified-id="Python-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Python</a></span></li><li><span><a href="#Why-Python-and-not-'X'-(put-any-other-language-here)" data-toc-modified-id="Why-Python-and-not-'X'-(put-any-other-language-here)-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Why Python and not 'X' (put any other language here)</a></span></li><li><span><a href="#Popularity-of-Python-for-teaching" data-toc-modified-id="Popularity-of-Python-for-teaching-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Popularity of Python for teaching</a></span></li><li><span><a href="#Python-ecosystem-for-scientific-computing-(main-libraries)" data-toc-modified-id="Python-ecosystem-for-scientific-computing-(main-libraries)-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Python ecosystem for scientific computing (main libraries)</a></span></li><li><span><a href="#The-Jupyter-Notebook" data-toc-modified-id="The-Jupyter-Notebook-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>The Jupyter Notebook</a></span></li><li><span><a href="#Jupyter-Notebook-and-IPython-kernel-architectures" data-toc-modified-id="Jupyter-Notebook-and-IPython-kernel-architectures-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Jupyter Notebook and IPython kernel architectures</a></span></li><li><span><a href="#Installing-the-Python-ecosystem" data-toc-modified-id="Installing-the-Python-ecosystem-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Installing the Python ecosystem</a></span><ul class="toc-item"><li><span><a href="#Anaconda" data-toc-modified-id="Anaconda-13.1"><span class="toc-item-num">13.1&nbsp;&nbsp;</span>Anaconda</a></span></li><li><span><a href="#Miniconda" data-toc-modified-id="Miniconda-13.2"><span class="toc-item-num">13.2&nbsp;&nbsp;</span>Miniconda</a></span></li></ul></li><li><span><a href="#IDE-for-Python" data-toc-modified-id="IDE-for-Python-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>IDE for Python</a></span></li><li><span><a href="#To-learn-about-Python" data-toc-modified-id="To-learn-about-Python-15"><span class="toc-item-num">15&nbsp;&nbsp;</span>To learn about Python</a></span></li><li><span><a href="#More-examples-of-Jupyter-Notebooks" data-toc-modified-id="More-examples-of-Jupyter-Notebooks-16"><span class="toc-item-num">16&nbsp;&nbsp;</span>More examples of Jupyter Notebooks</a></span></li><li><span><a href="#Questions?" data-toc-modified-id="Questions?-17"><span class="toc-item-num">17&nbsp;&nbsp;</span>Questions?</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The [Python programming language](https://www.python.org/) with [its ecosystem for scientific programming](https://scipy.org/) has features, maturity, and a community of developers and users that makes it the ideal environment for the scientific community.   

        This talk will show some of these features and usage examples.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Computing as a third kind of Science

        Traditionally, science has been divided into experimental and theoretical disciplines, but nowadays computing plays an important role in science. Scientific computation is sometimes related to theory, and at other times to experimental work. Hence, it is often seen as a new third branch of science.

        <figure><img src="https://raw.githubusercontent.com/jrjohansson/scientific-python-lectures/master/images/theory-experiment-computation.png" width=300 alt="theory-experiment-computation"/></figure>  

        Figure from [J.R. Johansson](http://nbviewer.jupyter.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-0-Scientific-Computing-with-Python.ipynb).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## About Python [[Python documentation](http://www.python.org/doc/essays/blurb/)]

        *Python is a programming language that lets you work more quickly and integrate your systems more effectively. You can learn to use Python and see almost immediate gains in productivity and lower maintenance costs* [[python.org](http://python.org/)].
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - *Python is an interpreted, object-oriented, high-level programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together*.  
        - *Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost of program maintenance. Python supports modules and packages, which encourages program modularity and code reuse*.  
        - Python is free and open source.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## About Python [[Python documentation](http://www.python.org/doc/essays/blurb/)]

        - *Often, programmers fall in love with Python because of the increased productivity it provides. Since there is no compilation step, the edit-test-debug cycle is incredibly fast. Debugging Python programs is easy: a bug or bad input will never cause a segmentation fault. Instead, when the interpreter discovers an error, it raises an exception. When the program doesn't catch the exception, the interpreter prints a stack trace.*  
        - A source level debugger allows inspection of local and global variables, evaluation of arbitrary expressions, setting breakpoints, stepping through the code a line at a time, and so on. The debugger is written in Python itself, testifying to Python's introspective power. On the other hand, often the quickest way to debug a program is to add a few print statements to the source: the fast edit-test-debug cycle makes this simple approach very effective.*
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Glossary for the Python technical characteristics I

         - Programming language: a formal language designed to communicate instructions to a computer. A sequence of instructions that specifies how to perform a computation is called a program.
         - Interpreted language: a program in an interpreted language is executed or interpreted by an interpreter program. This interpreter executes the program source code, statement by statement.
         - Compiled language: a program in a compiled language is first explicitly translated by the user into a lower-level machine language executable (with a compiler) and then this program can be executed.
         - Python interpreter: an interpreter is the computer program that executes the program. The most-widely used implementation of the Python programming language, referred as CPython or simply Python, is written in C (another programming language, which is lower-level and compiled).
         - High-level: a high-level programming language has a strong abstraction from the details of the computer and the language is independent of a particular type of computer. A high-level programming language is closer to human languages than to the programming language running inside the computer that communicate instructions to its hardware, the machine language. The machine language is a low-level programming language, in fact, the lowest one.
         - Object-oriented programming: a programming paradigm that represents concepts as "objects" that have data fields (attributes that describe the object) and associated procedures known as methods.
         - Semantics and syntax: the term semantics refers to the meaning of a language, as opposed to its form, the syntax.
         - Static and dynamic semantics: static and dynamic refer to the point in time at which some programming element is resolved. Static indicates that resolution takes place at the time a program is written. Dynamic indicates that resolution takes place at the time a program is executed.
         - Static and dynamic typing and binding: in dynamic typing, the type of the variable (e.g., if it is an integer or a string or a different type of element) is not explicitly declared, it can change, and in general is not known until execution time. In static typing, the type of the variable must be declared and it is known before the execution time.  
         - Rapid Application Development: a software development methodology that uses minimal planning in favor of rapid prototyping.  
         - Scripting: the writing of scripts, small pieces of simple instructions (programs) that can be rapidly executed.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Glossary for the Python technical characteristics II

         - Glue language: a programming language for writing programs to connect software components (including programs written in other programming languages).
         - Modules and packages: a module is a file containing Python definitions (e.g., functions) and statements. Packages are a way of structuring Python’s module namespace by using “dotted module names”. For example, the module name A.B designates a submodule named B in a package named A. To be used, modules and packages have to be imported in Python with the import function. Namespace is a container for a set of identifiers (names), and allows the disambiguation of homonym identifiers residing in different namespaces. For example, with the command `import math`, we will have all the functions and statements defined in this module in the namespace '`math.`', for example, `math.pi` is the$\pi$constant and `math.cos()`, the cosine function.
         - Program modularity and code reuse: the degree that programs can be compartmentalized (divided in smaller programs) to facilitate program reuse.
         - Source or binary form: source refers to the original code of the program (typically in a text format) which would need to be compiled to a binary form (not anymore human readable) to be able to be executed.
         - Major platforms: typically refers to the main operating systems (OS) in the market: Windows (by Microsoft), Mac OSX (by Apple), and Linux distributions (such as Debian, Ubuntu, Mint, etc.). Mac OSX and Linux distros are derived from, or heavily inspired by, another operating system called Unix.
         - Edit-test-debug cycle: the typical cycle in the life of a programmer; write (edit) the code, run (test) it, and correct errors or improve it (debug). The read–eval–print loop (REPL) is another related term.
         - Segmentation fault: an error in a program that is generated by the hardware which notifies the operating system about a memory access violation.
         - Exception: an error in a program detected during execution is called an exception and the Python interpreter raises a message about this error (an exception is not necessarily fatal, i.e., does not necessarily terminate or break the program).
         - Stack trace: information related to what caused the exception describing the line of the program where it occurred with a possible history of related events.
         - Source level debugger: Python has a module (named pdb) for interactive source code debugging.
         - Local and global variables: refers to the scope of the variables. A local variable is defined inside a function and typically can be accessed (it exists) only inside that function unless declared as global.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## About Python

        Python is also the name of the software with the most-widely used implementation of the language (maintained by the [Python Software Foundation](http://www.python.org/psf/)).  
        This implementation is written mostly in the *C* programming language and it is nicknamed CPython.  
        So, the following phrase is correct: download Python *(the software)* to program in Python *(the language)* because Python *(both)* is great!   
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Python

        The origin of the name for the Python language in fact is not because of the big snake, the author of the Python language, Guido van Rossum, named the language after Monty Python, a famous British comedy group in the 70's.  
        By coincidence, the Monty Python group was also interested in human movement science:
        """
    )
    return


@app.cell
def _():
    from IPython.display import YouTubeVideo
    YouTubeVideo('eCLp7zodUiI', width=480, height=360, rel=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Why Python and not 'X' (put any other language here)

        Python is not the best programming language for all needs and for all people. There is no such language.   
        Now, if you are doing scientific computing, chances are that Python is perfect for you because (and might also be perfect for lots of other needs):

        - Python is free, open source, and cross-platform.  
        - Python is easy to learn, with readable code, well documented, and with a huge and friendly user community.
        - Python is a real programming language, able to handle a variety of problems, easy to scale from small to huge problems, and easy to integrate with other systems (including other programming languages).
        - Python code is not the fastest but Python is one the fastest languages for programming. It is not uncommon in science to care more about the time we spend programming than the time the program took to run. But if code speed is important, one can easily integrate in different ways a code written in other languages (such as C and Fortran) with Python.
        - The Jupyter Notebook is a versatile tool for programming, data visualization, plotting, simulation, numeric and symbolic mathematics, and writing for daily use.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Popularity of Python for teaching
        """
    )
    return


@app.cell
def _():
    from IPython.display import IFrame
    IFrame('https://cacm.acm.org/blogs/blog-cacm/176450-python-is-now-the-most-popular-introductory-teaching-language-at-top-us-universities/fulltext',
           width=800, height=600)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Python ecosystem for scientific computing (main libraries)

        - [Python](https://www.python.org/) of course (the CPython distribution): a free, open source and cross-platform programming language that lets you work more quickly and integrate your systems more effectively.
        - [Numpy](https://numpy.org/): fundamental package for scientific computing with a N-dimensional array package.
        - [Scipy](https://scipy.org/): numerical routines for scientific computing.
        - [Matplotlib](https://matplotlib.org/): comprehensive 2D Plotting.
        - [Sympy](https://www.sympy.org/en/index.html): symbolic mathematics.
        - [Pandas](https://pandas.pydata.org/): data structures and data analysis tools.
        - [IPython](http://ipython.org): provides a rich architecture for interactive computing with powerful interactive shell, kernel for Jupyter, support for interactive data visualization and use of GUI toolkits, flexible embeddable interpreters, and high performance tools for parallel computing.  
        - [Jupyter Notebook](https://jupyter.org/): web application that allows you to create and share documents that contain live code, equations, visualizations and explanatory text.
        - [Statsmodels](https://www.statsmodels.org/stable/index.html#): to explore data, estimate statistical models, and perform statistical tests.
        - [Scikit-learn](https://scikit-learn.org/stable/): tools for data mining and data analysis (including machine learning).
        - [Pillow](https://python-pillow.org/): Python Imaging Library.
        - [Spyder](https://github.com/spyder-ide/spyder): interactive development environment with advanced editing, interactive testing, debugging and introspection features.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Jupyter Notebook

        The Jupyter Notebook App is a server-client application that allows editing and running notebook documents via a web browser. The Jupyter Notebook App can be executed on a local desktop requiring no Internet access (as described in this document) or installed on a remote server and accessed through the Internet.  

        Notebook documents (or “notebooks”, all lower case) are documents produced by the Jupyter Notebook App which contain both computer code (e.g. python) and rich text elements (paragraph, equations, figures, links, etc...). Notebook documents are both human-readable documents containing the analysis description and the results (figures, tables, etc..) as well as executable documents which can be run to perform data analysis.

        [Try Jupyter Notebook in your browser](https://try.jupyter.org/).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Jupyter Notebook and IPython kernel architectures

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/jupyternotebook.png?raw=1" width=800 alt="Jupyter Notebook and IPython kernel architectures"/></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Installing the Python ecosystem

        **The easy way**   
        The easiest way to get Python and the most popular packages for scientific programming is to install them with a Python distribution such as [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  

        In fact, you don't even need to install Python in your computer, you can run Python for scientific programming in the cloud using [python.org](https://www.python.org/shell/), [Google Colaboratory](https://colab.research.google.com/), or [repl.it](https://replit.com/languages/python3).

        **The hard way**   
        You can download Python and all individual packages you need and install them one by one. In general, it's not that difficult, but it can become challenging and painful for certain big packages heavily dependent on math, image visualization, and your operating system (i.e., Microsoft Windows).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Anaconda

        Go to the [*Anaconda* website](https://www.anaconda.com/products/distribution) and download the appropriate version for your computer (but download Anaconda3! for Python 3.x). The file is big (about 500 MB).

        Follow the installation steps described in the [Anaconda documentatione](https://docs.anaconda.com/anaconda/install/) for your operational system.   
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Miniconda

        A variation of *Anaconda* is [*Miniconda*](https://docs.conda.io/en/latest/miniconda.html) (Miniconda3 for Python 3.x), which contains only the *Conda* package manager and Python.  

        Once *Miniconda* is installed, you can use the `conda` command to install any other packages and create environments, etc.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # My current installation
        """
    )
    return


@app.cell
def _():
    import sys
    sys.version
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        More information can be obtained using the [watermark extension](https://github.com/rasbt/watermark):
        """
    )
    return


app._unparsable_cell(
    r"""
    try:
        from watermark import watermark
    except ImportError:
        %pip install -q watermark
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext watermark
    """,
    name="_"
)


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %watermark -u -t -d -m -v --iversions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## IDE for Python

        You might want an Integrated Development Environment (IDE) for programming in Python.  
        See [10 Best Python IDE & Code Editors](https://hackr.io/blog/best-python-ide) for possible IDEs.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## To learn about Python

        There is a lot of good material in the Internet about Python for scientific computing, some of them are:  

         - [How To Think Like A Computer Scientist](http://openbookproject.net/thinkcs/python/english3e/) or [the interactive edition](https://runestone.academy/ns/books/published/thinkcspy/index.html) (book)
         - [Python Scientific Lecture Notes](http://scipy-lectures.org/) (lecture notes)  
         - [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) (tutorial/book)    
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## More examples of Jupyter Notebooks

        Let's run stuff from:
        - [https://github.com/BMClab/BMC](https://github.com/BMClab/BMC)  
        - [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Questions?

        - https://www.reddit.com/r/learnpython/
        - https://stackoverflow.com/questions/tagged/python
        - https://www.reddit.com/r/Python/  
        """
    )
    return


@app.cell
def _():
    import this
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
