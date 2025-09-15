import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Matrix

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
        A matrix is a square or rectangular array of numbers or symbols (termed elements), arranged in rows and columns. For instance:$\mathbf{A} = 
        \begin{bmatrix} 
        a_{1,1} & a_{1,2} & a_{1,3} \\
        a_{2,1} & a_{2,2} & a_{2,3} 
        \end{bmatrix}$$\mathbf{A} = 
        \begin{bmatrix} 
        1 & 2 & 3 \\
        4 & 5 & 6 
        \end{bmatrix}$The matrix$\mathbf{A}$above has two rows and three columns, it is a 2x3 matrix.

        In Numpy:
        """
    )
    return


@app.cell
def _():
    # Import the necessary libraries
    import numpy as np
    from IPython.display import display
    np.set_printoptions(precision=4)  # number of digits of precision for floating point
    return (np,)


@app.cell
def _(np):
    A = np.array([[1, 2, 3], [4, 5, 6]])
    A
    return (A,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To get information about the number of elements and the structure of the matrix (in fact, a Numpy array), we can use:
        """
    )
    return


@app.cell
def _(A, np):
    print('A:\n', A)
    print('len(A) = ', len(A))
    print('np.size(A) = ', np.size(A))
    print('np.shape(A) = ', np.shape(A))
    print('np.ndim(A) = ', np.ndim(A))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We could also have accessed this information with the correspondent methods:
        """
    )
    return


@app.cell
def _(A):
    print('A.size = ', A.size)
    print('A.shape = ', A.shape)
    print('A.ndim = ', A.ndim)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We used the array function in Numpy to represent a matrix. A [Numpy array is in fact different than a matrix](http://www.scipy.org/NumPy_for_Matlab_Users), if we want to use explicit matrices in Numpy, we have to use the function `mat`:
        """
    )
    return


@app.cell
def _(np):
    B = np.mat([[1, 2, 3], [4, 5, 6]])
    B
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Both array and matrix types work in Numpy, but you should choose only one type and not mix them; the array is preferred because it is [the standard vector/matrix/tensor type of Numpy](http://www.scipy.org/NumPy_for_Matlab_Users). So, let's use the array type for the rest of this text.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Addition and multiplication

        The sum of two m-by-n matrices$\mathbf{A}$and$\mathbf{B}$is another m-by-n matrix:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""$\mathbf{A} = 
        \begin{bmatrix} 
        a_{1,1} & a_{1,2} & a_{1,3} \\
        a_{2,1} & a_{2,2} & a_{2,3} 
        \end{bmatrix}
        \;\;\; \text{and} \;\;\;
        \mathbf{B} =
        \begin{bmatrix} 
        b_{1,1} & b_{1,2} & b_{1,3} \\
        b_{2,1} & b_{2,2} & b_{2,3} 
        \end{bmatrix}$$\mathbf{A} + \mathbf{B} = 
        \begin{bmatrix} 
        a_{1,1}+b_{1,1} & a_{1,2}+b_{1,2} & a_{1,3}+b_{1,3} \\
        a_{2,1}+b_{2,1} & a_{2,2}+b_{2,2} & a_{2,3}+b_{2,3} 
        \end{bmatrix}$In Numpy:
        """
    )
    return


@app.cell
def _(np):
    A_1 = np.array([[1, 2, 3], [4, 5, 6]])
    B_1 = np.array([[7, 8, 9], [10, 11, 12]])
    print('A:\n', A_1)
    print('B:\n', B_1)
    print('A + B:\n', A_1 + B_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The multiplication of the m-by-n matrix$\mathbf{A}$by the n-by-p matrix$\mathbf{B}$is a m-by-p matrix:$\mathbf{A} = 
        \begin{bmatrix} 
        a_{1,1} & a_{1,2} \\
        a_{2,1} & a_{2,2} 
        \end{bmatrix}
        \;\;\; \text{and} \;\;\;
        \mathbf{B} =
        \begin{bmatrix} 
        b_{1,1} & b_{1,2} & b_{1,3} \\
        b_{2,1} & b_{2,2} & b_{2,3} 
        \end{bmatrix}$$\mathbf{A} \mathbf{B} = 
        \begin{bmatrix} 
        a_{1,1}b_{1,1} + a_{1,2}b_{2,1} & a_{1,1}b_{1,2} + a_{1,2}b_{2,2} & a_{1,1}b_{1,3} + a_{1,2}b_{2,3} \\
        a_{2,1}b_{1,1} + a_{2,2}b_{2,1} & a_{2,1}b_{1,2} + a_{2,2}b_{2,2} & a_{2,1}b_{1,3} + a_{2,2}b_{2,3}
        \end{bmatrix}$In Numpy:
        """
    )
    return


@app.cell
def _(np):
    A_2 = np.array([[1, 2], [3, 4]])
    B_2 = np.array([[5, 6, 7], [8, 9, 10]])
    print('A:\n', A_2)
    print('B:\n', B_2)
    print('A x B:\n', np.dot(A_2, B_2))
    return A_2, B_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that because the array type is not truly a matrix type, we used the dot product to calculate matrix multiplication.   
        We can use the matrix type to show the equivalent:
        """
    )
    return


@app.cell
def _(A_2, B_2, np):
    A_3 = np.mat(A_2)
    B_3 = np.mat(B_2)
    print('A:\n', A_3)
    print('B:\n', B_3)
    print('A x B:\n', A_3 * B_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Same result as before.

        The order in multiplication matters,$\mathbf{AB} \neq \mathbf{BA}$:
        """
    )
    return


@app.cell
def _(np):
    A_4 = np.array([[1, 2], [3, 4]])
    B_4 = np.array([[5, 6], [7, 8]])
    print('A:\n', A_4)
    print('B:\n', B_4)
    print('A x B:\n', np.dot(A_4, B_4))
    print('B x A:\n', np.dot(B_4, A_4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The addition or multiplication of a scalar (a single number) to a matrix is performed over all the elements of the matrix:
        """
    )
    return


@app.cell
def _(np):
    A_5 = np.array([[1, 2], [3, 4]])
    c = 10
    print('A:\n', A_5)
    print('c:\n', c)
    print('c + A:\n', c + A_5)
    print('cA:\n', c * A_5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Transposition

        The transpose of the matrix$\mathbf{A}$is the matrix$\mathbf{A^T}$turning all the rows of matrix$\mathbf{A}$into columns (or columns into rows):$\mathbf{A} = 
        \begin{bmatrix} 
        a & b & c \\
        d & e & f \end{bmatrix}
        \;\;\;\;\;\;\iff\;\;\;\;\;\;
        \mathbf{A^T} = 
        \begin{bmatrix} 
        a & d \\
        b & e \\
        c & f
        \end{bmatrix}$In NumPy, the transpose operator can be used as a method or function:
        """
    )
    return


@app.cell
def _(np):
    A_6 = np.array([[1, 2], [3, 4]])
    print('A:\n', A_6)
    print('A.T:\n', A_6.T)
    print('np.transpose(A):\n', np.transpose(A_6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Determinant

        The determinant is a number associated with a square matrix.

        The determinant of the following matrix:$\left[  \begin{array}{ccc}
        a & b & c \\
        d & e & f \\
        g & h & i \end{array} \right]$is written as:$\left|  \begin{array}{ccc}
        a & b & c \\
        d & e & f \\
        g & h & i \end{array} \right|$And has the value:$(aei + bfg + cdh) - (ceg + bdi + afh)$One way to manually calculate the determinant of a matrix is to use the [rule of Sarrus](http://en.wikipedia.org/wiki/Rule_of_Sarrus): we repeat the last columns (all columns but the first one) in the right side of the matrix and calculate the sum of the products of three diagonal north-west to south-east lines of matrix elements, minus the sum of the products of three diagonal south-west to north-east lines of elements as illustrated in the following figure:  
        <br>
        <figure><img src='http://upload.wikimedia.org/wikipedia/commons/6/66/Sarrus_rule.svg' width=300 alt='Rule of Sarrus'/><center><figcaption><i>Figure. Rule of Sarrus: the sum of the products of the solid diagonals minus the sum of the products of the dashed diagonals (<a href="http://en.wikipedia.org/wiki/Rule_of_Sarrus">image from Wikipedia</a>).</i></figcaption></center> </figure>

        In Numpy, the determinant is computed with the `linalg.det` function:
        """
    )
    return


@app.cell
def _(np):
    A_7 = np.array([[1, 2], [3, 4]])
    print('A:\n', A_7)
    return (A_7,)


@app.cell
def _(A_7, np):
    print('Determinant of A:\n', np.linalg.det(A_7))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Identity

        The identity matrix$\mathbf{I}$is a matrix with ones in the main diagonal and zeros otherwise. The 3x3 identity matrix is:$\mathbf{I} = 
        \begin{bmatrix} 
        1 & 0 & 0 \\
        0 & 1 & 0 \\
        0 & 0 & 1 \end{bmatrix}$In Numpy, instead of manually creating this matrix we can use the function `eye`:
        """
    )
    return


@app.cell
def _(np):
    np.eye(3)  # identity 3x3 array
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Inverse

        The inverse of the matrix$\mathbf{A}$is the matrix$\mathbf{A^{-1}}$such that the product between these two matrices is the identity matrix:$\mathbf{A}\cdot\mathbf{A^{-1}} = \mathbf{I}$The calculation of the inverse of a matrix is usually not simple (the inverse of the matrix$\mathbf{A}$is not$1/\mathbf{A}$; there is no division operation between matrices).  The Numpy function `linalg.inv` computes the inverse of a square matrix:   

            numpy.linalg.inv(a)
            Compute the (multiplicative) inverse of a matrix.
            Given a square matrix a, return the matrix ainv satisfying dot(a, ainv) = dot(ainv, a) = eye(a.shape[0]).
        """
    )
    return


@app.cell
def _(np):
    A_8 = np.array([[1, 2], [3, 4]])
    print('A:\n', A_8)
    _Ainv = np.linalg.inv(A_8)
    print('Inverse of A:\n', _Ainv)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Pseudo-inverse

        For a non-square matrix, its inverse is not defined. However, we can calculate what it's known as the pseudo-inverse.  
        Consider a non-square matrix,$\mathbf{A}$. To calculate its inverse, note that the following manipulation results in the identity matrix:$\mathbf{A} \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} = \mathbf{I}$The$\mathbf{A} \mathbf{A}^T$is a square matrix and is invertible (also [nonsingular](https://en.wikipedia.org/wiki/Invertible_matrix)) if$\mathbf{A}$is L.I. ([linearly independent rows/columns](https://en.wikipedia.org/wiki/Linear_independence)).  
        The matrix$\mathbf{A}^T(\mathbf{A}\mathbf{A}^T)^{-1}$is known as the [generalized inverse or Moore–Penrose pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse) of the matrix$\mathbf{A}$, a generalization of the inverse matrix.

        To compute the Moore–Penrose pseudoinverse, we could calculate it by a naive approach in Python:
        ```python
        from numpy.linalg import inv
        Ainv = A.T @ inv(A @ A.T)
        ```
        But both Numpy and Scipy have functions to calculate the pseudoinverse, which might give greater numerical stability (but read [Inverses and pseudoinverses. Numerical issues, speed, symmetry](http://vene.ro/blog/inverses-pseudoinverses-numerical-issues-speed-symmetry.html)). Of note, [numpy.linalg.pinv](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html) calculates the pseudoinverse of a matrix using its singular-value decomposition (SVD) and including all large singular values (using the [LAPACK (Linear Algebra Package)](https://en.wikipedia.org/wiki/LAPACK) routine gesdd), whereas [scipy.linalg.pinv](http://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pinv.html#scipy.linalg.pinv) calculates a pseudoinverse of a matrix using a least-squares solver (using the LAPACK method gelsd) and [scipy.linalg.pinv2](http://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pinv2.html) also uses SVD to find the pseudoinverse (also using the LAPACK routine gesdd).  

        For example:
        """
    )
    return


@app.cell
def _(np):
    from scipy.linalg import pinv2
    A_9 = np.array([[1, 0, 0], [0, 1, 0]])
    Apinv = pinv2(A_9)
    print('Matrix A:\n', A_9)
    print('Pseudo-inverse of A:\n', Apinv)
    print('A x Apinv:\n', A_9 @ Apinv)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Orthogonality

        A square matrix is said to be orthogonal if:

        1. There is no linear combination of one of the lines or columns of the matrix that would lead to the other row or column.   
        2. Its columns or rows form a basis of (independent) unit vectors (versors).

        As consequence:

        1. Its determinant is equal to 1 or -1.
        2. Its inverse is equal to its transpose.

        However, keep in mind that not all matrices with determinant equals to one are orthogonal, for example, the matrix:$\begin{bmatrix}
        3 & 2 \\
        4 & 3 
        \end{bmatrix}$Has determinant equals to one but it is not orthogonal (the columns or rows don't have norm equals to one).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Linear equations

        > A linear equation is an algebraic equation in which each term is either a constant or the product of a constant and (the first power of) a single variable ([Wikipedia](http://en.wikipedia.org/wiki/Linear_equation)).

        We are interested in solving a set of linear equations where two or more variables are unknown, for instance:$x + 2y = 4$$3x + 4y = 10$Let's see how to employ the matrix formalism to solve these equations (even that we know the solution is `x=2` and `y=1`).   
        Let's express this set of equations in matrix form:$\begin{bmatrix} 
        1 & 2 \\
        3 & 4 \end{bmatrix}
        \begin{bmatrix} 
        x \\
        y \end{bmatrix}
        = \begin{bmatrix} 
        4 \\
        10 \end{bmatrix}$And for the general case:$\mathbf{Av} = \mathbf{c}$Where$\mathbf{A, v, c}$are the matrices above and we want to find the values `x,y` for the matrix$\mathbf{v}$.   
        Because there is no division of matrices, we can use the inverse of$\mathbf{A}$to solve for$\mathbf{v}$:$\mathbf{A}^{-1}\mathbf{Av} = \mathbf{A}^{-1}\mathbf{c} \implies$$\mathbf{v} = \mathbf{A}^{-1}\mathbf{c}$As we know how to compute the inverse of$\mathbf{A}$, the solution is:
        """
    )
    return


@app.cell
def _(np):
    A_10 = np.array([[1, 2], [3, 4]])
    _Ainv = np.linalg.inv(A_10)
    c_1 = np.array([4, 10])
    _v = np.dot(_Ainv, c_1)
    print('v:\n', _v)
    return A_10, c_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        What we expected.

        However, the use of the inverse of a matrix to solve equations is computationally inefficient.   
        Instead, we should use `linalg.solve` for a determined system (same number of equations and unknowns) or `linalg.lstsq` otherwise:   
        From the help for `solve`:   

            numpy.linalg.solve(a, b)[source]
            Solve a linear matrix equation, or system of linear scalar equations.
            Computes the “exact” solution, x, of the well-determined, i.e., full rank, linear matrix equation ax = b.
        """
    )
    return


@app.cell
def _(A_10, c_1, np):
    _v = np.linalg.solve(A_10, c_1)
    print('Using solve:')
    print('v:\n', _v)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And from the help for `lstsq`:

            numpy.linalg.lstsq(a, b, rcond=-1)[source]
            Return the least-squares solution to a linear matrix equation.
            Solves the equation a x = b by computing a vector x that minimizes the Euclidean 2-norm || b - a x ||^2. The equation may be under-, well-, or over- determined (i.e., the number of linearly independent rows of a can be less than, equal to, or greater than its number of linearly independent columns). If a is square and of full rank, then x (but for round-off error) is the “exact” solution of the equation.
        """
    )
    return


@app.cell
def _(A_10, c_1, np):
    _v = np.linalg.lstsq(A_10, c_1)[0]
    print('Using lstsq:')
    print('v:\n', _v)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Same solutions, of course.

        When a system of equations has a unique solution, the determinant of the **square** matrix associated to this system of equations is nonzero.   
        When the determinant is zero there are either no solutions or many solutions to the system of equations.

        But if we have an overdetermined system:$x + 2y = 4$$3x + 4y = 10$$5x + 6y = 15$(Note that the possible solution for this set of equations is not exact because the last equation should be equal to 16.)

        Let's try to solve it:
        """
    )
    return


@app.cell
def _(np):
    A_11 = np.array([[1, 2], [3, 4], [5, 6]])
    print('A:\n', A_11)
    c_2 = np.array([4, 10, 15])
    print('c:\n', c_2)
    return A_11, c_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Because the matix$\mathbf{A}$is not squared, we can calculate its pseudo-inverse or use the function `linalg.lstsq`:
        """
    )
    return


@app.cell
def _(A_11, c_2, np):
    _v = np.linalg.lstsq(A_11, c_2)[0]
    print('Using lstsq:')
    print('v:\n', _v)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The functions `inv` and `solve` failed because the matrix$\mathbf{A}$was not square (overdetermined system). The function `lstsq` not only was able to handle an overdetermined system but was also able to find the best approximate solution.

        And if the the set of equations was undetermined, `lstsq` would also work. For instance, consider the system:$x + 2y + 2z = 10$$3x + 4y + z = 13$And in matrix form:$\begin{bmatrix} 
        1 & 2 & 2 \\
        3 & 4 & 1 \end{bmatrix}
        \begin{bmatrix} 
        x \\
        y \\
        z \end{bmatrix}
        = \begin{bmatrix} 
        10 \\
        13 \end{bmatrix}$A possible solution would be `x=2,y=1,z=3`, but other values would also satisfy this set of equations.

        Let's try to solve using `lstsq`:
        """
    )
    return


@app.cell
def _(np):
    A_12 = np.array([[1, 2, 2], [3, 4, 1]])
    print('A:\n', A_12)
    c_3 = np.array([10, 13])
    print('c:\n', c_3)
    return A_12, c_3


@app.cell
def _(A_12, c_3, np):
    _v = np.linalg.lstsq(A_12, c_3)[0]
    print('Using lstsq:')
    print('v:\n', _v)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is an approximated solution and as explained in the help of `solve`, this solution, `v`, is the one that minimizes the Euclidean norm$|| \mathbf{c - A v} ||^2$.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
