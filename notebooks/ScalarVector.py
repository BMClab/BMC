import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Scalar and vector

        > Marcos Duarte, Renato Naville Watanabe  
        > [Laboratory of Biomechanics and Motor Control](http://pesquisa.ufabc.edu.br/bmclab)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Python-setup" data-toc-modified-id="Python-setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Python setup</a></span></li><li><span><a href="#Scalar" data-toc-modified-id="Scalar-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Scalar</a></span><ul class="toc-item"><li><span><a href="#Scalar-operations-in-Python" data-toc-modified-id="Scalar-operations-in-Python-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Scalar operations in Python</a></span></li></ul></li><li><span><a href="#Vector" data-toc-modified-id="Vector-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Vector</a></span><ul class="toc-item"><li><span><a href="#Magnitude-(length-or-norm)-of-a-vector" data-toc-modified-id="Magnitude-(length-or-norm)-of-a-vector-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Magnitude (length or norm) of a vector</a></span></li><li><span><a href="#Vecton-addition-and-subtraction" data-toc-modified-id="Vecton-addition-and-subtraction-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Vecton addition and subtraction</a></span></li></ul></li><li><span><a href="#Dot-product" data-toc-modified-id="Dot-product-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Dot product</a></span></li><li><span><a href="#Vector-product" data-toc-modified-id="Vector-product-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Vector product</a></span><ul class="toc-item"><li><span><a href="#Gram–Schmidt-process" data-toc-modified-id="Gram–Schmidt-process-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Gram–Schmidt process</a></span></li></ul></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-Internet" data-toc-modified-id="Video-lectures-on-the-Internet-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Video lectures on the Internet</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Python handles very well all mathematical operations with numeric scalars and vectors and you can use [Sympy](http://sympy.org) for similar stuff but with abstract symbols. Let's briefly review scalars and vectors and show how to use Python for numerical calculation.  

        For a review about scalars and vectors, see chapter 2 of [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Python setup
        """
    )
    return


@app.cell
def _():
    from IPython.display import IFrame
    import math
    import numpy as np
    return IFrame, math, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Scalar

        >A **scalar** is a one-dimensional physical quantity, which can be described by a single real number.  
        For example, time, mass, and energy are examples of scalars.

        ### Scalar operations in Python

        Simple arithmetic operations with scalars are indeed simple:
        """
    )
    return


@app.cell
def _(math):
    a = 2
    b = 3
    print('a =', a, ', b =', b)
    print('a + b =', a + b)
    print('a - b =', a - b)
    print('a * b =', a * b)
    print('a / b =', a / b)
    print('a ** b =', a ** b)
    print('sqrt(b) =', math.sqrt(b))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you have a set of numbers, or an array, it is probably better to use Numpy; it will be faster for large data sets, and combined with Scipy, has many more mathematical funcions.
        """
    )
    return


@app.cell
def _(np):
    a_1 = 2
    b_1 = [3, 4, 5, 6, 7, 8]
    b_1 = np.array(b_1)
    print('a =', a_1, ', b =', b_1)
    print('a + b =', a_1 + b_1)
    print('a - b =', a_1 - b_1)
    print('a * b =', a_1 * b_1)
    print('a / b =', a_1 / b_1)
    print('a ** b =', a_1 ** b_1)
    print('np.sqrt(b) =', np.sqrt(b_1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Numpy performs the arithmetic operations of the single number in `a` with all the numbers of the array `b`. This is called broadcasting in computer science.   
        Even if you have two arrays (but they must have the same size), Numpy handles for you:
        """
    )
    return


@app.cell
def _(np):
    a_2 = np.array([1, 2, 3])
    b_2 = np.array([4, 5, 6])
    print('a =', a_2, ', b =', b_2)
    print('a + b =', a_2 + b_2)
    print('a - b =', a_2 - b_2)
    print('a * b =', a_2 * b_2)
    print('a / b =', a_2 / b_2)
    print('a ** b =', a_2 ** b_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Vector

        >A **vector** is a quantity with magnitude (or length) and direction expressed numerically as an ordered list of values according to a coordinate reference system.  
        For example, position, force, and torque are physical quantities defined by vectors.

        For instance, consider the position of a point in space represented by a vector:  
        <br>
        <figure><img src="./../images/vector3D.png" width=300/><figcaption><center><i>Figure. Position of a point represented by a vector in a Cartesian coordinate system.</i></center></figcaption></figure>  


        The position of the point (the vector) above can be represented as a tuple of values:$(x,\: y,\: z) \; \Rightarrow \; (1, 3, 2)$or in matrix form:$\begin{bmatrix} x \\y \\z \end{bmatrix} \;\; \Rightarrow  \;\; \begin{bmatrix} 1 \\3 \\2 \end{bmatrix}$We can use the Numpy array to represent the components of vectors.   
        For instance, for the vector above is expressed in Python as:
        """
    )
    return


@app.cell
def _(np):
    a_3 = np.array([1, 3, 2])
    print('a =', a_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exactly like the arrays in the last example for scalars, so all operations we performed will result in the same values, of course.   
        However, as we are now dealing with vectors, now some of the  operations don't make sense. For example, for vectors there are no multiplication, division, power, and square root in the way we calculated.

        A vector can also be represented as:

        <span class="notranslate">$\overrightarrow{\mathbf{a}} = a_x\hat{\mathbf{i}} + a_y\hat{\mathbf{j}} + a_z\hat{\mathbf{k}}$</span>
        <br>
        <figure><img src="./../images/vector3Dijk.png" width=300/><figcaption><center><i>Figure. A vector representation in a Cartesian coordinate system. The versors <span class="notranslate">$\hat{\mathbf{i}},\, \hat{\mathbf{j}},\, \hat{\mathbf{k}}\,$</span> are usually represented in the color sequence <b>rgb</b> (red, green, blue) for easier visualization.</i></center></figcaption></figure>

        Where <span class="notranslate">$\hat{\mathbf{i}},\, \hat{\mathbf{j}},\, \hat{\mathbf{k}}\,$</span> are unit vectors, each representing a direction and <span class="notranslate">$a_x\hat{\mathbf{i}},\: a_y\hat{\mathbf{j}},\: a_z\hat{\mathbf{k}}$</span> are the vector components of the vector <span class="notranslate">$\overrightarrow{\mathbf{a}}$</span>.

        A unit vector (or versor) is a vector whose length (or norm) is 1.   
        The unit vector of a non-zero vector <span class="notranslate">$\overrightarrow{\mathbf{a}}$</span> is the unit vector codirectional with <span class="notranslate">$\overrightarrow{\mathbf{a}}$</span>:

        <span class="notranslate">$\mathbf{\hat{u}} = \frac{\overrightarrow{\mathbf{a}}}{||\overrightarrow{\mathbf{a}}||} = \frac{a_x\,\hat{\mathbf{i}} + a_y\,\hat{\mathbf{j}} + a_z\, \hat{\mathbf{k}}}{\sqrt{a_x^2+a_y^2+a_z^2}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Magnitude (length or norm) of a vector

        The magnitude (length) of a vector is often represented by the symbol$||\;||$, also known as the norm (or Euclidean norm) of a vector and it is defined as:

        <span class="notranslate">$||\overrightarrow{\mathbf{a}}|| = \sqrt{a_x^2+a_y^2+a_z^2}$</span>
        The function `numpy.linalg.norm` calculates the norm:
        """
    )
    return


@app.cell
def _(np):
    a_4 = np.array([1, 2, 3])
    np.linalg.norm(a_4)
    return (a_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Or we can use the definition and compute directly:
        """
    )
    return


@app.cell
def _(a_4, np):
    np.sqrt(np.sum(a_4 * a_4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Then, the versor for the vector <span class="notranslate">$\overrightarrow{\mathbf{a}} = (1, 2, 3)$</span> is:
        """
    )
    return


@app.cell
def _(np):
    a_5 = np.array([1, 2, 3])
    u = a_5 / np.linalg.norm(a_5)
    print('u =', u)
    return (u,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And we can verify its magnitude is indeed 1:
        """
    )
    return


@app.cell
def _(np, u):
    np.linalg.norm(u)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        But the representation of a vector as a tuple of values is only valid for a vector with its origin coinciding with the origin$(0, 0, 0)$of the coordinate system we adopted.
        For instance, consider the following vector:  
        <br>
        <figure><img src="./../images/vector2.png" width=260/><figcaption><center><i>Figure. A vector in space.</i></center></figcaption></figure>

        Such a vector cannot be represented by <span class="notranslate">$(b_x, b_y, b_z)$</span> because this would be for the vector from the origin to the point B. To represent exactly this vector we need the two vectors <span class="notranslate">$\mathbf{a}$</span> and <span class="notranslate">$\mathbf{b}$</span>. This fact is important when we perform some calculations in Mechanics.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Vecton addition and subtraction
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The addition of two vectors is another vector:
        <span class="notranslate">$\overrightarrow{\mathbf{a}} + \overrightarrow{\mathbf{b}} = (a_x\hat{\mathbf{i}} + a_y\hat{\mathbf{j}} + a_z\hat{\mathbf{k}}) + (b_x\hat{\mathbf{i}} + b_y\hat{\mathbf{j}} + b_z\hat{\mathbf{k}}) = 
        (a_x+b_x)\hat{\mathbf{i}} + (a_y+b_y)\hat{\mathbf{j}} + (a_z+b_z)\hat{\mathbf{k}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <figure><img src="http://upload.wikimedia.org/wikipedia/commons/2/28/Vector_addition.svg" width=300 alt="Vector addition"/><figcaption><center><i>Figure. Vector addition (image from Wikipedia).</i></center></figcaption></figure> 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The subtraction of two vectors is also another vector:

        <span class="notranslate">$\overrightarrow{\mathbf{a}} - \overrightarrow{\mathbf{b}} = (a_x\hat{\mathbf{i}} + a_y\hat{\mathbf{j}} + a_z\hat{\mathbf{k}}) + (b_x\hat{\mathbf{i}} + b_y\hat{\mathbf{j}} + b_z\hat{\mathbf{k}}) = 
        (a_x-b_x)\hat{\mathbf{i}} + (a_y-b_y)\hat{\mathbf{j}} + (a_z-b_z)\hat{\mathbf{k}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <figure><img src="http://upload.wikimedia.org/wikipedia/commons/2/24/Vector_subtraction.svg" width=160 alt="Vector subtraction"/><figcaption><center><i>Figure. Vector subtraction (image from Wikipedia).</i></center></figcaption></figure></div>  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Consider two 2D arrays (rows and columns) representing the position of two objects moving in space. The columns represent the vector components and the rows the values of the position vector in different instants.   
        Once again, it's easy to perform addition and subtraction with these vectors:
        """
    )
    return


@app.cell
def _(np):
    a_6 = np.array([[1, 2, 3], [1, 1, 1]])
    b_3 = np.array([[4, 5, 6], [7, 8, 9]])
    print('a =', a_6, '\nb =', b_3)
    print('a + b =', a_6 + b_3)
    print('a - b =', a_6 - b_3)
    return (a_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Numpy can handle a N-dimensional array with the size limited by the available memory in your computer.

        And we can perform operations on each vector, for example, calculate the norm of each one.   
        First let's check the shape of the variable `a` using the method `shape` or the function `numpy.shape`:
        """
    )
    return


@app.cell
def _(a_6, np):
    print(a_6.shape)
    print(np.shape(a_6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This means the variable `a` has 2 rows and 3 columns.   
        We have to tell the function `numpy.norm` to calculate the norm for each vector, i.e., to operate through the columns of the variable `a` using the paraneter `axis`:
        """
    )
    return


@app.cell
def _(a_6, np):
    np.linalg.norm(a_6, axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Dot product

        Dot product (or scalar product or inner product) between two vectors is a mathematical operation algebraically defined as the sum of the products of the corresponding components (maginitudes in each direction) of the two vectors. The result of the dot product is a single number (a scalar).  
        The dot product between vectors <span class="notranslate">$\overrightarrow{\mathbf{a}}$</span> and$\overrightarrow{\mathbf{b}}$is:

        <span class="notranslate">$\overrightarrow{\mathbf{a}} \cdot \overrightarrow{\mathbf{b}} = (a_x\,\hat{\mathbf{i}}+a_y\,\hat{\mathbf{j}}+a_z\,\hat{\mathbf{k}}) \cdot (b_x\,\hat{\mathbf{i}}+b_y\,\hat{\mathbf{j}}+b_z\,\hat{\mathbf{k}}) = a_x b_x + a_y b_y + a_z b_z$</span>

        Because by definition:

        <span class="notranslate">$\hat{\mathbf{i}} \cdot \hat{\mathbf{i}} = \hat{\mathbf{j}} \cdot \hat{\mathbf{j}} = \hat{\mathbf{k}} \cdot \hat{\mathbf{k}}= 1 \quad \text{and} \quad \hat{\mathbf{i}} \cdot \hat{\mathbf{j}} = \hat{\mathbf{i}} \cdot \hat{\mathbf{k}} = \hat{\mathbf{j}} \cdot \hat{\mathbf{k}} = 0$</span>

        The geometric equivalent of the dot product is the product of the magnitudes of the two vectors and the cosine of the angle between them:

        <span class="notranslate">$\overrightarrow{\mathbf{a}} \cdot \overrightarrow{\mathbf{b}} = ||\overrightarrow{\mathbf{a}}||\:||\overrightarrow{\mathbf{b}}||\:cos(\theta)$</span>


        Which is also equivalent to state that the dot product between two vectors$\overrightarrow{\mathbf{a}}$and$\overrightarrow{\mathbf{b}}$is the magnitude of$\overrightarrow{\mathbf{a}}$times the magnitude of the component of$\overrightarrow{\mathbf{b}}$parallel to$\overrightarrow{\mathbf{a}}$(or the magnitude of$\overrightarrow{\mathbf{b}}$times the magnitude of the component of$\overrightarrow{\mathbf{a}}$parallel to$\overrightarrow{\mathbf{b}}$).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The dot product between two vectors can be visualized in this interactive animation:
        """
    )
    return


@app.cell
def _(IFrame):
    IFrame('https://www.geogebra.org/classic/ncdf2jsw?embed',
           width='100%', height=500)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The Numpy function for the dot product is `numpy.dot`:
        """
    )
    return


@app.cell
def _(np):
    a_7 = np.array([1, 2, 3])
    b_4 = np.array([4, 5, 6])
    print('a =', a_7, '\nb =', b_4)
    print('np.dot(a, b) =', np.dot(a_7, b_4))
    return a_7, b_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Or we can use the definition and compute directly:
        """
    )
    return


@app.cell
def _(a_7, b_4, np):
    np.sum(a_7 * b_4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For 2D arrays, the `numpy.dot` function performs matrix multiplication rather than the dot product; so let's use the `numpy.sum` function:
        """
    )
    return


@app.cell
def _(np):
    a_8 = np.array([[1, 2, 3], [1, 1, 1]])
    b_5 = np.array([[4, 5, 6], [7, 8, 9]])
    np.sum(a_8 * b_5, axis=1)
    return a_8, b_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Vector product

        Cross product or vector product between two vectors is a mathematical operation in three-dimensional space which results in a vector perpendicular to both of the vectors being multiplied and a length (norm) equal to the product of the perpendicular components of the vectors being multiplied (which is equal to the area of the parallelogram that the vectors span).   
        The cross product between vectors <span class="notranslate">$\overrightarrow{\mathbf{a}}$</span> and <span class="notranslate">$\overrightarrow{\mathbf{b}}$</span> is:

        <span class="notranslate">$\overrightarrow{\mathbf{a}} \times \overrightarrow{\mathbf{b}} = (a_x\,\hat{\mathbf{i}} + a_y\,\hat{\mathbf{j}} + a_z\,\hat{\mathbf{k}}) \times (b_x\,\hat{\mathbf{i}}+b_y\,\hat{\mathbf{j}}+b_z\,\hat{\mathbf{k}}) = (a_yb_z-a_zb_y)\hat{\mathbf{i}} + (a_zb_x-a_xb_z)\hat{\mathbf{j}}+(a_xb_y-a_yb_x)\hat{\mathbf{k}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Because by definition:

        <span class="notranslate">$\begin{array}{l l}
        \hat{\mathbf{i}} \times \hat{\mathbf{i}} = \hat{\mathbf{j}} \times \hat{\mathbf{j}} = \hat{\mathbf{k}} \times \hat{\mathbf{k}} = 0 \\
        \hat{\mathbf{i}} \times \hat{\mathbf{j}} = \hat{\mathbf{k}}, \quad \hat{\mathbf{k}} \times \hat{\mathbf{k}} = \hat{\mathbf{i}}, \quad \hat{\mathbf{k}} \times \hat{\mathbf{i}} = \hat{\mathbf{j}} \\
        \hat{\mathbf{j}} \times \hat{\mathbf{i}} = -\hat{\mathbf{k}}, \quad \hat{\mathbf{k}} \times \hat{\mathbf{j}}= -\hat{\mathbf{i}}, \quad \hat{\mathbf{i}} \times \hat{\mathbf{k}} = -\hat{\mathbf{j}}
        \end{array}$</span>

        The direction of the vector resulting from the cross product between the vectors <span class="notranslate">$\overrightarrow{\mathbf{a}}$</span> and <span class="notranslate">$\overrightarrow{\mathbf{b}}$</span> is given by the right-hand rule.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The geometric equivalent of the cross product is:

        The geometric equivalent of the cross product is the product of the magnitudes of the two vectors and the sine of the angle between them:

        <span class="notranslate">$\overrightarrow{\mathbf{a}} \times \overrightarrow{\mathbf{b}} = ||\overrightarrow{\mathbf{a}}||\:||\overrightarrow{\mathbf{b}}||\:sin(\theta)$</span>

        Which is also equivalent to state that the cross product between two vectors <span class="notranslate">$\overrightarrow{\mathbf{a}}$</span> and <span class="notranslate">$\overrightarrow{\mathbf{b}}$</span> is the magnitude of <span class="notranslate">$\overrightarrow{\mathbf{a}}$</span> times the magnitude of the component of <span class="notranslate">$\overrightarrow{\mathbf{b}}$</span> perpendicular to <span class="notranslate">$\overrightarrow{\mathbf{a}}$</span> (or the magnitude of <span class="notranslate">$\overrightarrow{\mathbf{b}}$</span> times the magnitude of the component of <span class="notranslate">$\overrightarrow{\mathbf{a}}$</span> perpendicular to <span class="notranslate">$\overrightarrow{\mathbf{b}}$</span>).

        The definition above, also implies that the magnitude of the cross product is the area of the parallelogram spanned by the two vectors:  
        <br>
        <figure><img src="http://upload.wikimedia.org/wikipedia/commons/4/4e/Cross_product_parallelogram.svg" width=160 alt="Vector subtraction"/><figcaption><center><i>Figure. Area of a parallelogram as the magnitude of the cross product (image from Wikipedia).</i></center></figcaption></figure> 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The cross product can also be calculated as the determinant of a matrix:

        <span class="notranslate">$\overrightarrow{\mathbf{a}} \times \overrightarrow{\mathbf{b}} = \left| \begin{array}{ccc}
        \hat{\mathbf{i}} & \hat{\mathbf{j}} & \hat{\mathbf{k}} \\
        a_x & a_y & a_z \\
        b_x & b_y & b_z 
        \end{array} \right|
        = a_y b_z \hat{\mathbf{i}} + a_z b_x \hat{\mathbf{j}} +  a_x b_y \hat{\mathbf{k}} - a_y b_x \hat{\mathbf{k}}-a_z b_y \hat{\mathbf{i}} - a_x b_z \hat{\mathbf{j}} \\
        \overrightarrow{\mathbf{a}} \times \overrightarrow{\mathbf{b}} = (a_yb_z-a_zb_y)\hat{\mathbf{i}} + (a_zb_x-a_xb_z)\hat{\mathbf{j}} + (a_xb_y-a_yb_x)\hat{\mathbf{k}}$</span>

        The same result as before.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The cross product between two vectors can be visualized in this interactive animation:
        """
    )
    return


@app.cell
def _(IFrame):
    IFrame('https://www.geogebra.org/classic/cz6v2U99?embed',
           width='100%', height=500)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The Numpy function for the cross product is `numpy.cross`:
        """
    )
    return


@app.cell
def _(a_8, b_5, np):
    print('a =', a_8, '\nb =', b_5)
    print('np.cross(a, b) =', np.cross(a_8, b_5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For 2D arrays with vectors in different rows:
        """
    )
    return


@app.cell
def _(np):
    a_9 = np.array([[1, 2, 3], [1, 1, 1]])
    b_6 = np.array([[4, 5, 6], [7, 8, 9]])
    np.cross(a_9, b_6, axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Gram–Schmidt process

        The [Gram–Schmidt process](http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) is a method for orthonormalizing (orthogonal unit versors) a set of vectors using the scalar product. The Gram–Schmidt process works for any number of vectors.   
        For example, given three vectors, <span class="notranslate">$\overrightarrow{\mathbf{a}}, \overrightarrow{\mathbf{b}}, \overrightarrow{\mathbf{c}}$</span>, in the 3D space, a basis <span class="notranslate">$\{\hat{e}_a, \hat{e}_b, \hat{e}_c\}$</span> can be found using the Gram–Schmidt process by: 

        The first versor is in the <span class="notranslate">$\overrightarrow{\mathbf{a}}$</span> direction (or in the direction of any of the other vectors):  

        <span class="notranslate">$\hat{e}_a = \frac{\overrightarrow{\mathbf{a}}}{||\overrightarrow{\mathbf{a}}||}$</span>

        The second versor, orthogonal to <span class="notranslate">$\hat{e}_a$</span>, can be found considering we can express vector <span class="notranslate">$\overrightarrow{\mathbf{b}}$</span> in terms of the <span class="notranslate">$\hat{e}_a$</span> direction as:  

        <span class="notranslate">$\overrightarrow{\mathbf{b}} = \overrightarrow{\mathbf{b}}^\| + \overrightarrow{\mathbf{b}}^\bot$</span>

        Then:

        <span class="notranslate">$\overrightarrow{\mathbf{b}}^\bot = \overrightarrow{\mathbf{b}} - \overrightarrow{\mathbf{b}}^\| = \overrightarrow{\mathbf{b}} - (\overrightarrow{\mathbf{b}} \cdot \hat{e}_a ) \hat{e}_a$</span>

        Finally:

        <span class="notranslate">$\hat{e}_b = \frac{\overrightarrow{\mathbf{b}}^\bot}{||\overrightarrow{\mathbf{b}}^\bot||}$</span>

        The third versor, orthogonal to <span class="notranslate">$\{\hat{e}_a, \hat{e}_b\}$</span>, can be found expressing the vector <span class="notranslate">$\overrightarrow{\mathbf{C}}$</span> in terms of <span class="notranslate">$\hat{e}_a$</span> and <span class="notranslate">$\hat{e}_b$</span> directions as:

        <span class="notranslate">$\overrightarrow{\mathbf{c}} = \overrightarrow{\mathbf{c}}^\| + \overrightarrow{\mathbf{c}}^\bot$</span>

        Then:

        <span class="notranslate">$\overrightarrow{\mathbf{c}}^\bot = \overrightarrow{\mathbf{c}} - \overrightarrow{\mathbf{c}}^\|$</span>

        Where:

        <span class="notranslate">$\overrightarrow{\mathbf{c}}^\| = (\overrightarrow{\mathbf{c}} \cdot \hat{e}_a ) \hat{e}_a + (\overrightarrow{\mathbf{c}} \cdot \hat{e}_b ) \hat{e}_b$</span>

        Finally:

        <span class="notranslate">$\hat{e}_c = \frac{\overrightarrow{\mathbf{c}}^\bot}{||\overrightarrow{\mathbf{c}}^\bot||}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's implement the Gram–Schmidt process in Python.

        For example, consider the positions (vectors) <span class="notranslate">$\overrightarrow{\mathbf{a}} = [1,2,0], \overrightarrow{\mathbf{b}} = [0,1,3], \overrightarrow{\mathbf{c}} = [1,0,1]$</span>:
        """
    )
    return


@app.cell
def _(np):
    a_10 = np.array([1, 2, 0])
    b_7 = np.array([0, 1, 3])
    c = np.array([1, 0, 1])
    return a_10, b_7, c


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The first versor is:
        """
    )
    return


@app.cell
def _(a_10, np):
    ea = a_10 / np.linalg.norm(a_10)
    print(ea)
    return (ea,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The second versor is:
        """
    )
    return


@app.cell
def _(b_7, ea, np):
    eb = b_7 - np.dot(b_7, ea) * ea
    eb = eb / np.linalg.norm(eb)
    print(eb)
    return (eb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the third version is:
        """
    )
    return


@app.cell
def _(c, ea, eb, np):
    ec = c - np.dot(c, ea)*ea - np.dot(c, eb)*eb
    ec = ec/np.linalg.norm(ec)
    print(ec)
    return (ec,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's check the orthonormality between these versors:
        """
    )
    return


@app.cell
def _(ea, eb, ec, np):
    print('Versors:', '\nea =', ea, '\neb =', eb, '\nec =', ec)
    print('\nTest of orthogonality (scalar product between versors):',
          '\nea x eb:', np.dot(ea, eb),
          '\neb x ec:', np.dot(eb, ec),
          '\nec x ea:', np.dot(ec, ea))
    print('\nNorm of each versor:',
          '\n||ea|| =', np.linalg.norm(ea),
          '\n||eb|| =', np.linalg.norm(eb),
          '\n||ec|| =', np.linalg.norm(ec))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Or, we can simply use the built-in QR factorization function from NumPy:
        """
    )
    return


@app.cell
def _(a_10, b_7, c, np):
    vectors = np.vstack((a_10, b_7, c)).T
    (Q, R) = np.linalg.qr(vectors)
    print(Q)
    return (Q,)


@app.cell
def _(Q, np):
    (ea_1, eb_1, ec_1) = (Q[:, 0], Q[:, 1], Q[:, 2])
    print('Versors:', '\nea =', ea_1, '\neb =', eb_1, '\nec =', ec_1)
    print('\nTest of orthogonality (scalar product between versors):')
    print(np.dot(Q.T, Q))
    print('\nTest of orthogonality (scalar product between versors):', '\nea x eb:', np.dot(ea_1, eb_1), '\neb x ec:', np.dot(eb_1, ec_1), '\nec x ea:', np.dot(ec_1, ea_1))
    print('\nNorm of each versor:', '\n||ea|| =', np.linalg.norm(ea_1), '\n||eb|| =', np.linalg.norm(eb_1), '\n||ec|| =', np.linalg.norm(ec_1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Which results in the same basis with exception of the changed signals.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

         - Read pages 44-92 of the first chapter of the [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html) about scalars and vectors in Mechanics.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the Internet

         - Khan Academy: [Vectors](https://www.khanacademy.org/math/algebra-home/alg-vectors)  
         - [Vectors, what even are they?](https://youtu.be/fNk_zzaMoSs)  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. Given the vectors, <span class="notranslate">$\overrightarrow{\mathbf{a}}=[1, 0, 0]$</span> and <span class="notranslate">$\overrightarrow{\mathbf{b}}=[1, 1, 1]$</span>, calculate the dot and cross products between them.  

        2. Calculate the unit vectors for$[2, −2, 3]$and$[3, −3, 2]$and determine an orthogonal vector to these two vectors.  

        3. Given the vectors <span class="notranslate">$\overrightarrow{\mathbf{a}}$=[1, 0, 0]</span> and <span class="notranslate">$\overrightarrow{\mathbf{b}}$=[1, 1, 1], calculate$\overrightarrow{\mathbf{a}} \times \overrightarrow{\mathbf{b}}$</span> and verify that this vector is orthogonal to vectors <span class="notranslate">$\overrightarrow{\mathbf{a}}$</span> and <span class="notranslate">$\overrightarrow{\mathbf{b}}$</span>. Also, calculate <span class="notranslate">$\overrightarrow{\mathbf{b}} \times \overrightarrow{\mathbf{a}}$</span> and compare it with <span class="notranslate">$\overrightarrow{\mathbf{a}} \times \overrightarrow{\mathbf{b}}$</span>.  

        4. Given the vectors$[1, 1, 0]; [1, 0, 1]; [0, 1, 1]$, calculate a basis using the Gram–Schmidt process.

        5. Write a Python function to calculate a basis using the Gram–Schmidt process (implement the algorithm!) considering that the input are three variables where each one contains the coordinates of vectors as columns and different positions of these vectors as rows. For example, sample variables can be generated with the command `np.random.randn(5, 3)`. 

        6. Study the sample problems **1.1** to **1.9**, **1.11** (using Python), **1.12**, **1.14**, **1.17**, **1.18** to **1.24** of Ruina and Rudra's book

        7. From Ruina and Rudra's book, solve the problems **1.1.1** to **1.3.16**. 

        If you are new to scalars and vectors, you should solve these problems first by hand and then use Python to check the answers.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Ruina A, Rudra P (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.  
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
