import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scalar and vector

    > Marcos Duarte, Renato Naville Watanabe,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A motion capture system hands you three numbers per marker per frame. A force plate hands you three more. Almost everything you will ever compute in biomechanics starts from triples like those, and almost every mistake you will make with them comes from forgetting that a triple of numbers is not automatically a vector, and that a vector is not automatically something you are allowed to multiply.

    This notebook is a tour of the small set of operations that *are* allowed — magnitude, addition, dot product, cross product — and of what each of them means physically. It ends by building a coordinate system out of three measured points, which is the operation that all three-dimensional motion analysis rests on.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Python is used throughout as a calculator you can argue with: when a result surprises you, that is the notebook working.

    For a review of scalars and vectors, see chapter 1 of [Ruina and Pratap's book](http://ruina.tam.cornell.edu/Book/index.html). For symbolic rather than numerical work with the same objects, see [Sympy](http://sympy.org).

    **Challenge 0.** Before you begin, write down three physical quantities you have measured or would like to measure. For each, say whether one number is enough to describe it. Keep the list nearby; the checkpoint questions ask you to come back to it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three numbers from a laboratory

    Suppose a motion capture system reports the velocity of a jumper's center of mass at the instant of take-off as

    $$
    \overrightarrow{\mathbf{v}} = (8.0,\; 3.5,\; 0.2)\;\mathrm{m/s},
    $$

    where $\mathbf{x}$ is the direction of the run-up, $\mathbf{y}$ is vertical, and $\mathbf{z}$ is medio-lateral.

    Three questions you might reasonably ask of those numbers:

    - How fast was the jumper going? That is one number, and it is *not* 8.0, nor 11.7.
    - At what angle did they leave the ground? That is also one number, and there is more than one defensible way to compute it.
    - Is the 0.2 in the third slot noise, or is the jumper drifting sideways?

    None of the three can be answered by looking at the components one at a time. By the end of this notebook all three will be one line of Python each — and you will know which of the two angles in the second question you actually wanted.

    **Guiding questions 0.**

    1. Guess the magnitude of that velocity before computing it. Is it closer to 8, to 9, or to 12 m/s?
    2. The third component is small. Under what circumstance would a small number in that slot be the most interesting thing in the measurement?
    3. If you rotated the laboratory's coordinate system by 30 degrees about the vertical, which of the three numbers would change? Which of the answers to the three questions above would change?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy does all the work in this notebook. The `math` module appears once, to make a point about the difference between the two.
    """)
    return


@app.cell
def _():
    import math
    import numpy as np

    return math, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scalar

    > A **scalar** is a one-dimensional physical quantity, which can be described by a single real number.

    Time, mass, and energy are examples of scalars. So are temperature, distance travelled, and the reading on a bathroom scale.

    ### Scalar operations in Python

    Simple arithmetic operations with scalars are indeed simple:
    """)
    return


@app.cell
def _(math):
    a = 2
    b = 3

    print("a =", a, ", b =", b)
    print("a + b =", a + b)
    print("a - b =", a - b)
    print("a * b =", a * b)
    print("a / b =", a / b)
    print("a ** b =", a**b)
    print("sqrt(b) =", math.sqrt(b))
    return (a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you have a set of numbers, or an array, it is probably better to use NumPy; it will be faster for large data sets and, combined with SciPy, has many more mathematical functions.

    Note the last line below: `math.sqrt` would raise an error on an array, while `numpy.sqrt` takes the square root of every element at once. Use NumPy's functions on NumPy's arrays.
    """)
    return


@app.cell
def _(a, np):
    b_array = np.array([3, 4, 5, 6, 7, 8])

    print("a =", a, ", b =", b_array)
    print("a + b =", a + b_array)
    print("a - b =", a - b_array)
    print("a * b =", a * b_array)
    print("a / b =", a / b_array)
    print("a ** b =", a**b_array)
    print("np.sqrt(b) =", np.sqrt(b_array))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    NumPy performed the arithmetic operation between the single number in `a` and every number of the array `b`. This is called **broadcasting**, and it is the reason NumPy code so rarely contains a loop.

    Even with two arrays — provided their shapes are compatible — NumPy handles it for you:
    """)
    return


@app.cell
def _(np):
    c_array = np.array([1, 2, 3])
    d_array = np.array([4, 5, 6])

    print("c =", c_array, ", d =", d_array)
    print("c + d =", c_array + d_array)
    print("c - d =", c_array - d_array)
    print("c * d =", c_array * d_array)
    print("c / d =", c_array / d_array)
    print("c ** d =", c_array**d_array)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every one of those six lines ran without complaint. Hold that thought for two cells; three of them are about to become nonsense.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vector

    > A **vector** is a quantity with magnitude (or length) and direction, expressed numerically as an ordered list of values according to a coordinate reference system.

    Position, force, and torque are physical quantities defined by vectors. So is the take-off velocity we started with.

    For instance, consider the position of a point in space represented by a vector:

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/vector3D.png?raw=1" width=300 alt="position vector"/></center><figcaption><center><i>Figure. Position of a point represented by a vector in a Cartesian coordinate system.</i></center></figcaption></figure>

    The position of the point (the vector) above can be represented as a tuple of values:

    $$
    (x,\: y,\: z) \; \Rightarrow \; (1, 3, 2)
    $$

    or in matrix form:

    $$
    \begin{bmatrix} x \\ y \\ z \end{bmatrix} \;\; \Rightarrow \;\; \begin{bmatrix} 1 \\ 3 \\ 2 \end{bmatrix}
    $$

    We can use a NumPy array to represent the components of a vector. For the vector above:
    """)
    return


@app.cell
def _(np):
    p = np.array([1, 3, 2])

    print("p =", p)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is exactly the same kind of object as the arrays in the scalar section, so all the operations we performed there will run and produce numbers. But we are now dealing with vectors, and some of those operations no longer *mean* anything. **For vectors there is no multiplication, division, power, or square root in the element-by-element way we just computed them.** NumPy will not stop you; mechanics will.

    There are two products defined for vectors, and neither of them is `a * b`. We come to both below.

    **Challenge 1.** Take two position vectors, $(1, 3, 2)$ and $(2, 0, 1)$, and compute their element-wise product in Python. You will get three numbers. Invent a physical interpretation for them. Then explain why your interpretation falls apart if you rotate the coordinate system — and notice that this is precisely the test that the two real vector products pass.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Versors and the ijk notation

    A vector can also be written as a sum of contributions along each axis:

    $$
    \overrightarrow{\mathbf{a}} = a_x\hat{\mathbf{i}} + a_y\hat{\mathbf{j}} + a_z\hat{\mathbf{k}}
    $$

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/vector3Dijk.png?raw=1" width=300 alt="vector components"/></center><figcaption><center><i>Figure. A vector representation in a Cartesian coordinate system. The versors $\hat{\mathbf{i}},\, \hat{\mathbf{j}},\, \hat{\mathbf{k}}\,$ are usually represented in the color sequence <b>rgb</b> (red, green, blue) for easier visualization.</i></center></figcaption></figure>

    where $\hat{\mathbf{i}},\, \hat{\mathbf{j}},\, \hat{\mathbf{k}}\,$ are unit vectors, each representing a direction, and $a_x\hat{\mathbf{i}},\: a_y\hat{\mathbf{j}},\: a_z\hat{\mathbf{k}}$ are the vector components of the vector $\overrightarrow{\mathbf{a}}$.

    A **unit vector** (or **versor**) is a vector whose length (or norm) is 1. The unit vector of a non-zero vector $\overrightarrow{\mathbf{a}}$ is the unit vector codirectional with $\overrightarrow{\mathbf{a}}$:

    $$
    \mathbf{\hat{u}} = \frac{\overrightarrow{\mathbf{a}}}{||\overrightarrow{\mathbf{a}}||} =
    \frac{a_x\,\hat{\mathbf{i}} + a_y\,\hat{\mathbf{j}} + a_z\, \hat{\mathbf{k}}}{\sqrt{a_x^2+a_y^2+a_z^2}}
    $$

    A versor is how you separate *where something points* from *how much of it there is* — the direction of a force from its intensity, the orientation of a segment from its length. Nearly every three-dimensional calculation in biomechanics has a versor in it somewhere.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Magnitude (length or norm) of a vector

    The magnitude (length) of a vector is often represented by the symbol $||\;||$, also known as the norm (or Euclidean norm) of a vector, and it is defined as:

    $$
    ||\overrightarrow{\mathbf{a}}|| = \sqrt{a_x^2+a_y^2+a_z^2}
    $$

    The function `numpy.linalg.norm` calculates the norm:
    """)
    return


@app.cell
def _(np):
    r = np.array([1, 2, 3])

    np.linalg.norm(r)
    return (r,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or we can use the definition and compute it directly — the element-wise product is legitimate here because we immediately sum it, which is the dot product in disguise:
    """)
    return


@app.cell
def _(np, r):
    np.sqrt(np.sum(r * r))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then the versor for the vector $\overrightarrow{\mathbf{r}} = (1, 2, 3)$ is:
    """)
    return


@app.cell
def _(np, r):
    r_hat = r / np.linalg.norm(r)

    print("r_hat =", r_hat)
    return (r_hat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we can verify its magnitude is indeed 1:
    """)
    return


@app.cell
def _(np, r_hat):
    np.linalg.norm(r_hat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now answer the first of the three questions from the top of the notebook. How fast was the jumper going?
    """)
    return


@app.cell
def _(np):
    v_takeoff = np.array([8.0, 3.5, 0.2])  # m/s

    print(f"Take-off velocity: {v_takeoff} m/s")
    print(f"Take-off speed:    {np.linalg.norm(v_takeoff):.2f} m/s")
    print(f"Direction:         {v_takeoff / np.linalg.norm(v_takeoff)}")
    return (v_takeoff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The speed is 8.73 m/s: more than the 8.0 of the horizontal component alone, and far less than the 11.7 you would get by adding the three numbers. A norm is not a sum, and a vector is emphatically not the sum of its parts.

    Note also the vocabulary. **Speed** is the magnitude of the velocity — a scalar. **Velocity** is the vector. English uses the two words loosely; mechanics does not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Vectors that do not start at the origin

    The representation of a vector as a tuple of values is only valid for a vector with its origin coinciding with the origin $(0, 0, 0)$ of the coordinate system we adopted. For instance, consider the following vector:

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/vector2.png?raw=1" width=260 alt="a vector in space"/></center><figcaption><center><i>Figure. A vector in space.</i></center></figcaption></figure>

    Such a vector cannot be represented by $(b_x, b_y, b_z)$, because that would be the vector from the origin to the point B. To represent exactly this vector we need the two vectors $\mathbf{a}$ and $\mathbf{b}$. This fact is important when we perform some calculations in mechanics.

    This is where a surprising number of errors in motion analysis come from. A marker trajectory is a position vector, tied to the origin of the laboratory. A segment — the forearm, say — is *not* a position vector; it is the difference between two of them. Writing `elbow` where you meant `wrist - elbow` produces code that runs perfectly and answers a question nobody asked.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Vector addition and subtraction

    The addition of two vectors is another vector:

    $$
    \overrightarrow{\mathbf{a}} + \overrightarrow{\mathbf{b}} =
    (a_x\hat{\mathbf{i}} + a_y\hat{\mathbf{j}} + a_z\hat{\mathbf{k}}) + (b_x\hat{\mathbf{i}} + b_y\hat{\mathbf{j}} + b_z\hat{\mathbf{k}}) =
    (a_x+b_x)\hat{\mathbf{i}} + (a_y+b_y)\hat{\mathbf{j}} + (a_z+b_z)\hat{\mathbf{k}}
    $$

    <figure><center><img src="http://upload.wikimedia.org/wikipedia/commons/2/28/Vector_addition.svg" width=300 alt="Vector addition"/></center><figcaption><center><i>Figure. Vector addition (image from Wikipedia).</i></center></figcaption></figure>

    The subtraction of two vectors is also another vector:

    $$
    \overrightarrow{\mathbf{a}} - \overrightarrow{\mathbf{b}} =
    (a_x\hat{\mathbf{i}} + a_y\hat{\mathbf{j}} + a_z\hat{\mathbf{k}}) - (b_x\hat{\mathbf{i}} + b_y\hat{\mathbf{j}} + b_z\hat{\mathbf{k}}) =
    (a_x-b_x)\hat{\mathbf{i}} + (a_y-b_y)\hat{\mathbf{j}} + (a_z-b_z)\hat{\mathbf{k}}
    $$

    <figure><center><img src="http://upload.wikimedia.org/wikipedia/commons/2/24/Vector_subtraction.svg" width=160 alt="Vector subtraction"/></center><figcaption><center><i>Figure. Vector subtraction (image from Wikipedia).</i></center></figcaption></figure>

    Both of these you have already used without naming them. Adding vectors is what you do to two forces acting on the same body. Subtracting them is what you do to two positions to get a displacement, which is where the notebook [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb) begins.

    Consider two 2D arrays (rows and columns) representing the position of two objects moving in space. The columns represent the vector components and the rows the values of the position vector at different instants. Once again, it is easy to perform addition and subtraction with these vectors:
    """)
    return


@app.cell
def _(np):
    pos_a = np.array([[1, 2, 3], [1, 1, 1]])
    pos_b = np.array([[4, 5, 6], [7, 8, 9]])

    print("pos_a =", pos_a, "\npos_b =", pos_b)
    print("pos_a + pos_b =", pos_a + pos_b)
    print("pos_a - pos_b =", pos_a - pos_b)
    return pos_a, pos_b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    NumPy can handle an N-dimensional array with the size limited by the available memory in your computer. That "rows are instants, columns are coordinates" layout is how essentially every motion capture file you will read is organized, so it is worth getting comfortable with it now.

    We can perform operations on each vector — for example, calculate the norm of each one. First let's check the shape of the variable `pos_a`, using the method `shape` or the function `numpy.shape`:
    """)
    return


@app.cell
def _(np, pos_a):
    print(pos_a.shape)
    print(np.shape(pos_a))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This means the variable `pos_a` has 2 rows and 3 columns.

    We have to tell the function `numpy.linalg.norm` to calculate the norm for each vector — that is, to operate along the columns of `pos_a` — using the parameter `axis`:
    """)
    return


@app.cell
def _(np, pos_a):
    np.linalg.norm(pos_a, axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two numbers out, one per instant, which is what we wanted. Get `axis` wrong and you get one number out, or three, and both are meaningless — but neither raises an error.

    **Challenge 2.** Run the cell above again with `axis=0` and with no `axis` at all. Write down what each result is the norm *of*. Then decide which of the three you would most likely produce by accident in real code, and how you would catch it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dot product

    The **dot product** (or scalar product, or inner product) between two vectors is a mathematical operation algebraically defined as the sum of the products of the corresponding components of the two vectors. The result of the dot product is a single number — a scalar.

    The dot product between vectors $\overrightarrow{\mathbf{a}}$ and $\overrightarrow{\mathbf{b}}$ is:

    $$
    \overrightarrow{\mathbf{a}} \cdot \overrightarrow{\mathbf{b}} =
    (a_x\,\hat{\mathbf{i}}+a_y\,\hat{\mathbf{j}}+a_z\,\hat{\mathbf{k}}) \cdot (b_x\,\hat{\mathbf{i}}+b_y\,\hat{\mathbf{j}}+b_z\,\hat{\mathbf{k}}) =
    a_x b_x + a_y b_y + a_z b_z
    $$

    Because by definition:

    $$
    \hat{\mathbf{i}} \cdot \hat{\mathbf{i}} = \hat{\mathbf{j}} \cdot \hat{\mathbf{j}} = \hat{\mathbf{k}} \cdot \hat{\mathbf{k}}= 1
    \quad \text{and} \quad
    \hat{\mathbf{i}} \cdot \hat{\mathbf{j}} = \hat{\mathbf{i}} \cdot \hat{\mathbf{k}} = \hat{\mathbf{j}} \cdot \hat{\mathbf{k}} = 0
    $$

    The geometric equivalent of the dot product is the product of the magnitudes of the two vectors and the cosine of the angle between them:

    $$
    \overrightarrow{\mathbf{a}} \cdot \overrightarrow{\mathbf{b}} = ||\overrightarrow{\mathbf{a}}||\:||\overrightarrow{\mathbf{b}}||\:\cos(\theta)
    $$

    which is also equivalent to stating that the dot product between two vectors is the magnitude of $\overrightarrow{\mathbf{a}}$ times the magnitude of the component of $\overrightarrow{\mathbf{b}}$ parallel to $\overrightarrow{\mathbf{a}}$ (or the magnitude of $\overrightarrow{\mathbf{b}}$ times the magnitude of the component of $\overrightarrow{\mathbf{a}}$ parallel to $\overrightarrow{\mathbf{b}}$).

    That second reading is the one to keep. **The dot product measures how much of one vector lies along another.** It is zero exactly when the two are perpendicular, largest when they are parallel, and negative when they oppose each other — which is why it turns up in the definition of mechanical work, where only the part of the force along the displacement counts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dot product between two vectors can be visualized in this interactive animation. Drag one vector until the product reads zero, and note the configuration:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        """<iframe src="https://www.geogebra.org/classic/ncdf2jsw?embed"
        width="100%" height="500" style="border:none"></iframe>"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The NumPy function for the dot product is `numpy.dot`:
    """)
    return


@app.cell
def _(np):
    vec_a = np.array([1, 2, 3])
    vec_b = np.array([4, 5, 6])

    print("vec_a =", vec_a, "\nvec_b =", vec_b)
    print("np.dot(vec_a, vec_b) =", np.dot(vec_a, vec_b))
    return vec_a, vec_b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or we can use the definition and compute it directly:
    """)
    return


@app.cell
def _(np, vec_a, vec_b):
    np.sum(vec_a * vec_b)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For 2D arrays, the `numpy.dot` function performs matrix multiplication rather than the dot product; so let's use the `numpy.sum` function with the `axis` parameter:
    """)
    return


@app.cell
def _(np, pos_a, pos_b):
    np.sum(pos_a * pos_b, axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The angle of the jump, two ways

    We can now answer the second question from the top of the notebook. Inverting the geometric definition of the dot product gives the angle between any two vectors:

    $$
    \theta = \arccos\left(\frac{\overrightarrow{\mathbf{a}} \cdot \overrightarrow{\mathbf{b}}}
    {||\overrightarrow{\mathbf{a}}||\;||\overrightarrow{\mathbf{b}}||}\right)
    $$

    **Before you run the next cell**, predict the take-off angle of a jumper whose velocity is $(8.0, 3.5, 0.2)$ m/s. And predict whether the two ways of computing it below will agree.
    """)
    return


@app.cell
def _(np, v_takeoff):
    x_axis = np.array([1, 0, 0])

    _cos = np.dot(v_takeoff, x_axis) / (
        np.linalg.norm(v_takeoff) * np.linalg.norm(x_axis)
    )
    angle_3d = np.rad2deg(np.arccos(_cos))
    angle_sagittal = np.rad2deg(np.arctan2(v_takeoff[1], v_takeoff[0]))

    print(f"Angle with the x axis (in 3D):      {angle_3d:.2f}°")
    print(f"Angle in the sagittal plane (x, y): {angle_sagittal:.2f}°")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two disagree, slightly, and the disagreement is not a bug.

    The dot product with the $\mathbf{x}$ axis gives the angle between the velocity and that axis *in space* — a single angle that knows nothing about which plane the deviation happened in. The sagittal-plane calculation throws the medio-lateral component away first and asks for the angle of what remains, which is the quantity a long jump analysis actually wants. With a $z$ component of only 0.2 m/s the difference is a few hundredths of a degree; raise that component to 2 m/s and see what happens.

    This is the habit worth forming: an angle is always *between two specific things*, and if you cannot say which two, you cannot say what your number means.

    **Guiding questions 1.**

    1. Under what condition would the two angles above be exactly equal?
    2. The dot product of the velocity with the vertical axis $\hat{\mathbf{j}}$ is 3.5. What is the physical meaning of that number?
    3. A force and a displacement are perpendicular. How much work was done? Which property of the dot product did you just use?
    4. Why did we use `arctan2` rather than `arctan` in the second calculation? Try it with a jumper moving backwards, $(-8.0, 3.5, 0.2)$, and see.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vector product

    The **cross product**, or vector product, between two vectors is a mathematical operation in three-dimensional space which results in a vector perpendicular to both of the vectors being multiplied, with a length (norm) equal to the product of the perpendicular components of the vectors being multiplied — which is equal to the area of the parallelogram that the vectors span.

    The cross product between vectors $\overrightarrow{\mathbf{a}}$ and $\overrightarrow{\mathbf{b}}$ is:

    $$
    \overrightarrow{\mathbf{a}} \times \overrightarrow{\mathbf{b}} =
    (a_x\,\hat{\mathbf{i}} + a_y\,\hat{\mathbf{j}} + a_z\,\hat{\mathbf{k}}) \times (b_x\,\hat{\mathbf{i}}+b_y\,\hat{\mathbf{j}}+b_z\,\hat{\mathbf{k}}) =
    (a_yb_z-a_zb_y)\hat{\mathbf{i}} + (a_zb_x-a_xb_z)\hat{\mathbf{j}}+(a_xb_y-a_yb_x)\hat{\mathbf{k}}
    $$

    Because by definition:

    $$
    \begin{array}{l l}
    \hat{\mathbf{i}} \times \hat{\mathbf{i}} = \hat{\mathbf{j}} \times \hat{\mathbf{j}} = \hat{\mathbf{k}} \times \hat{\mathbf{k}} = 0 \\
    \hat{\mathbf{i}} \times \hat{\mathbf{j}} = \hat{\mathbf{k}}, \quad \hat{\mathbf{j}} \times \hat{\mathbf{k}} = \hat{\mathbf{i}}, \quad \hat{\mathbf{k}} \times \hat{\mathbf{i}} = \hat{\mathbf{j}} \\
    \hat{\mathbf{j}} \times \hat{\mathbf{i}} = -\hat{\mathbf{k}}, \quad \hat{\mathbf{k}} \times \hat{\mathbf{j}}= -\hat{\mathbf{i}}, \quad \hat{\mathbf{i}} \times \hat{\mathbf{k}} = -\hat{\mathbf{j}}
    \end{array}
    $$

    The direction of the vector resulting from the cross product is given by the right-hand rule. Note the second and third rows: **the cross product does not commute.** Swapping the operands flips the result. If you have ever got a joint moment with the wrong sign, this is usually why.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The geometric equivalent of the cross product is the product of the magnitudes of the two vectors and the sine of the angle between them:

    $$
    ||\overrightarrow{\mathbf{a}} \times \overrightarrow{\mathbf{b}}|| = ||\overrightarrow{\mathbf{a}}||\:||\overrightarrow{\mathbf{b}}||\:\sin(\theta)
    $$

    which is also equivalent to stating that the magnitude of the cross product between $\overrightarrow{\mathbf{a}}$ and $\overrightarrow{\mathbf{b}}$ is the magnitude of $\overrightarrow{\mathbf{a}}$ times the magnitude of the component of $\overrightarrow{\mathbf{b}}$ perpendicular to $\overrightarrow{\mathbf{a}}$ (or the magnitude of $\overrightarrow{\mathbf{b}}$ times the magnitude of the component of $\overrightarrow{\mathbf{a}}$ perpendicular to $\overrightarrow{\mathbf{b}}$).

    So the two products are complementary: the dot product keeps the parallel part and the cross product keeps the perpendicular one. Where the dot product vanishes, the cross product is largest, and the other way around.

    The definition above also implies that the magnitude of the cross product is the area of the parallelogram spanned by the two vectors:

    <figure><center><img src="http://upload.wikimedia.org/wikipedia/commons/4/4e/Cross_product_parallelogram.svg" width=160 alt="cross product parallelogram"/></center><figcaption><center><i>Figure. Area of a parallelogram as the magnitude of the cross product (image from Wikipedia).</i></center></figcaption></figure>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cross product can also be calculated as the determinant of a matrix:

    $$
    \overrightarrow{\mathbf{a}} \times \overrightarrow{\mathbf{b}} = \left| \begin{array}{ccc}
    \hat{\mathbf{i}} & \hat{\mathbf{j}} & \hat{\mathbf{k}} \\
    a_x & a_y & a_z \\
    b_x & b_y & b_z
    \end{array} \right|
    = a_y b_z \hat{\mathbf{i}} + a_z b_x \hat{\mathbf{j}} + a_x b_y \hat{\mathbf{k}} - a_y b_x \hat{\mathbf{k}} - a_z b_y \hat{\mathbf{i}} - a_x b_z \hat{\mathbf{j}}
    $$

    $$
    \overrightarrow{\mathbf{a}} \times \overrightarrow{\mathbf{b}} = (a_yb_z-a_zb_y)\hat{\mathbf{i}} + (a_zb_x-a_xb_z)\hat{\mathbf{j}} + (a_xb_y-a_yb_x)\hat{\mathbf{k}}
    $$

    The same result as before. This is the form worth memorizing, because it is the one you can reconstruct under pressure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cross product between two vectors can be visualized in this interactive animation:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        """<iframe src="https://www.geogebra.org/classic/cz6v2U99?embed"
        width="100%" height="500" style="border:none"></iframe>"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The NumPy function for the cross product is `numpy.cross`:
    """)
    return


@app.cell
def _(np, vec_a, vec_b):
    print("vec_a =", vec_a, "\nvec_b =", vec_b)
    print("np.cross(vec_a, vec_b) =", np.cross(vec_a, vec_b))
    print("np.cross(vec_b, vec_a) =", np.cross(vec_b, vec_a))
    print(
        "np.dot(vec_a, np.cross(vec_a, vec_b)) =",
        np.dot(vec_a, np.cross(vec_a, vec_b)),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The last two lines are the two defining properties, checked numerically: reversing the operands negates the result, and the result is perpendicular to both inputs, so its dot product with either of them is zero.

    For 2D arrays with vectors in different rows:
    """)
    return


@app.cell
def _(np, pos_a, pos_b):
    np.cross(pos_a, pos_b, axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Challenge 3.** A force $\overrightarrow{\mathbf{F}} = (0, -300, 0)$ N is applied at a point whose position relative to a joint centre is $\overrightarrow{\mathbf{r}} = (0.25, 0, 0)$ m. The moment of that force about the joint is $\overrightarrow{\mathbf{M}} = \overrightarrow{\mathbf{r}} \times \overrightarrow{\mathbf{F}}$. Compute it. Then compute $\overrightarrow{\mathbf{F}} \times \overrightarrow{\mathbf{r}}$ and say, in words, what going about it the wrong way round claims physically.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gram–Schmidt process

    Here is the payoff, and the reason the rest of this notebook exists.

    In three-dimensional motion analysis you place markers on a segment — three of them, say, on the thigh — and you need a *coordinate system* attached to that segment: three mutually perpendicular unit vectors that move with the bone. What you have is three points, in no particular arrangement, measured with error. What you need is an orthonormal basis. The [Gram–Schmidt process](http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) is how you get from one to the other, using nothing but the dot product and the norm.

    Given three vectors $\overrightarrow{\mathbf{a}}, \overrightarrow{\mathbf{b}}, \overrightarrow{\mathbf{c}}$ in 3D space, a basis $\{\hat{e}_a, \hat{e}_b, \hat{e}_c\}$ can be found as follows. The first versor is in the $\overrightarrow{\mathbf{a}}$ direction (or in the direction of any of the other vectors):

    $$
    \hat{e}_a = \frac{\overrightarrow{\mathbf{a}}}{||\overrightarrow{\mathbf{a}}||}
    $$

    The second versor, orthogonal to $\hat{e}_a$, can be found considering that we can express vector $\overrightarrow{\mathbf{b}}$ in terms of a part along $\hat{e}_a$ and a part perpendicular to it:

    $$
    \overrightarrow{\mathbf{b}} = \overrightarrow{\mathbf{b}}^\| + \overrightarrow{\mathbf{b}}^\bot
    $$

    Then, subtracting the parallel part — which the dot product hands us:

    $$
    \overrightarrow{\mathbf{b}}^\bot = \overrightarrow{\mathbf{b}} - \overrightarrow{\mathbf{b}}^\| = \overrightarrow{\mathbf{b}} - (\overrightarrow{\mathbf{b}} \cdot \hat{e}_a ) \hat{e}_a
    $$

    Finally:

    $$
    \hat{e}_b = \frac{\overrightarrow{\mathbf{b}}^\bot}{||\overrightarrow{\mathbf{b}}^\bot||}
    $$

    The third versor, orthogonal to both $\hat{e}_a$ and $\hat{e}_b$, is found the same way, expressing $\overrightarrow{\mathbf{c}}$ in terms of the two directions already fixed:

    $$
    \overrightarrow{\mathbf{c}} = \overrightarrow{\mathbf{c}}^\| + \overrightarrow{\mathbf{c}}^\bot,
    \qquad
    \overrightarrow{\mathbf{c}}^\| = (\overrightarrow{\mathbf{c}} \cdot \hat{e}_a ) \hat{e}_a + (\overrightarrow{\mathbf{c}} \cdot \hat{e}_b ) \hat{e}_b
    $$

    $$
    \overrightarrow{\mathbf{c}}^\bot = \overrightarrow{\mathbf{c}} - \overrightarrow{\mathbf{c}}^\|,
    \qquad
    \hat{e}_c = \frac{\overrightarrow{\mathbf{c}}^\bot}{||\overrightarrow{\mathbf{c}}^\bot||}
    $$

    The process works for any number of vectors, and the pattern never changes: take the next vector, subtract off everything that lies along the directions you already have, and normalize what is left.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's implement the Gram–Schmidt process in Python. Read the three vectors below as the positions of three markers:

    $$
    \overrightarrow{\mathbf{m_1}} = [1,2,0], \quad \overrightarrow{\mathbf{m_2}} = [0,1,3], \quad \overrightarrow{\mathbf{m_3}} = [1,0,1]
    $$
    """)
    return


@app.cell
def _(np):
    m1 = np.array([1, 2, 0])
    m2 = np.array([0, 1, 3])
    m3 = np.array([1, 0, 1])
    return m1, m2, m3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first versor is simply the first vector, normalized:
    """)
    return


@app.cell
def _(m1, np):
    e1 = m1 / np.linalg.norm(m1)

    print(e1)
    return (e1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The second versor is the second vector with its $\hat{e}_1$ component removed, then normalized:
    """)
    return


@app.cell
def _(e1, m2, np):
    e2 = m2 - np.dot(m2, e1) * e1
    e2 = e2 / np.linalg.norm(e2)

    print(e2)
    return (e2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And the third versor removes both components already accounted for:
    """)
    return


@app.cell
def _(e1, e2, m3, np):
    e3 = m3 - np.dot(m3, e1) * e1 - np.dot(m3, e2) * e2
    e3 = e3 / np.linalg.norm(e3)

    print(e3)
    return (e3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Never trust a basis you have not tested. Orthonormal means two things — every pair perpendicular (dot product zero) and every vector of unit length — so check both:
    """)
    return


@app.cell
def _(e1, e2, e3, np):
    print("Versors:", "\ne1 =", e1, "\ne2 =", e2, "\ne3 =", e3)
    print(
        "\nTest of orthogonality (dot product between versors):",
        "\ne1 . e2:", np.dot(e1, e2),
        "\ne2 . e3:", np.dot(e2, e3),
        "\ne3 . e1:", np.dot(e3, e1),
    )
    print(
        "\nNorm of each versor:",
        "\n||e1|| =", np.linalg.norm(e1),
        "\n||e2|| =", np.linalg.norm(e2),
        "\n||e3|| =", np.linalg.norm(e3),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dot products are not exactly zero and the norms are not exactly one; they are zero and one to about fifteen decimal places. That is floating-point arithmetic, not a failure of the method, and it is why orthogonality is tested with a tolerance rather than with `==`.

    Or we can simply use the built-in QR factorization function from NumPy, which performs an equivalent orthogonalization:
    """)
    return


@app.cell
def _(m1, m2, m3, np):
    markers = np.vstack((m1, m2, m3)).T
    Q, R = np.linalg.qr(markers)

    print(Q)
    return (Q,)


@app.cell
def _(Q, np):
    q1, q2, q3 = Q[:, 0], Q[:, 1], Q[:, 2]

    print("Versors:", "\nq1 =", q1, "\nq2 =", q2, "\nq3 =", q3)
    print("\nTest of orthogonality (the matrix Q' Q should be the identity):")
    print(np.dot(Q.T, Q))
    print(
        "\nNorm of each versor:",
        "\n||q1|| =", np.linalg.norm(q1),
        "\n||q2|| =", np.linalg.norm(q2),
        "\n||q3|| =", np.linalg.norm(q3),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This results in the same basis, with the exception of the changed signs — QR is free to choose the opposite direction along each axis, and it does. For a coordinate system attached to a body segment, that sign is the difference between an axis pointing forward and one pointing backward, so it is never a detail you can ignore.

    **Challenge 4.** The basis you built depends on the *order* of the three vectors: $\hat{e}_1$ is exactly along $\overrightarrow{\mathbf{m_1}}$, while $\overrightarrow{\mathbf{m_3}}$ only gets whatever direction is left over. Redo the process with the markers in the reverse order and compare the two bases. Then ask the question that matters in practice: if one marker is noisier than the other two, where in the ordering should it go?

    How these bases become anatomical frames of reference — and what to do when the markers move relative to the bone — is the subject of the notebook [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to the three quantities you listed in Challenge 0. Which are scalars, which are vectors, and was any of them harder to classify than you expected?
    2. You have two arrays of the same shape holding a force and a velocity, sampled over time. Write down the expression for the instantaneous power at every instant. Which product did you use, and why not the other one?
    3. Name a quantity whose magnitude you would report and whose direction you would discard, and a quantity for which discarding the direction would be malpractice.
    4. Why can the dot product be computed for vectors of any dimension, while the cross product as defined here is a strictly three-dimensional operation?
    5. Your code computes the angle between two segments and returns 1.05. What is missing from that statement, and what are the two most likely mistakes behind the number?
    6. Three markers on a rigid segment give you a basis. What would you check, frame by frame, to convince yourself the segment really is rigid?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Given the vectors $\overrightarrow{\mathbf{a}}=[1, 0, 0]$ and $\overrightarrow{\mathbf{b}}=[1, 1, 1]$, calculate the dot and cross products between them.

    2. Calculate the unit vectors for $[2, -2, 3]$ and $[3, -3, 2]$ and determine a vector orthogonal to these two vectors.

    3. Given the vectors $\overrightarrow{\mathbf{a}}=[1, 0, 0]$ and $\overrightarrow{\mathbf{b}}=[1, 1, 1]$, calculate $\overrightarrow{\mathbf{a}} \times \overrightarrow{\mathbf{b}}$ and verify that this vector is orthogonal to vectors $\overrightarrow{\mathbf{a}}$ and $\overrightarrow{\mathbf{b}}$. Also, calculate $\overrightarrow{\mathbf{b}} \times \overrightarrow{\mathbf{a}}$ and compare it with $\overrightarrow{\mathbf{a}} \times \overrightarrow{\mathbf{b}}$.

    4. Given the vectors $[1, 1, 0]$, $[1, 0, 1]$ and $[0, 1, 1]$, calculate a basis using the Gram–Schmidt process.

    5. Write a Python function to calculate a basis using the Gram–Schmidt process (implement the algorithm!), considering that the input consists of three variables, each containing the coordinates of vectors as columns and different positions of these vectors as rows. For example, sample variables can be generated with the command `np.random.randn(5, 3)`.

    6. A jumper leaves the ground with a velocity of $(8.0, 3.5, 0.2)$ m/s, as in this notebook. Using only the operations defined here, calculate: the component of that velocity along the direction $(1, 0, 1)/\sqrt{2}$; the angle between the velocity and the vertical; and the area of the parallelogram spanned by the velocity and the vertical versor. State what, if anything, each of the three means physically.

    7. Study the sample problems **1.1** to **1.9**, **1.11** (using Python), **1.12**, **1.14**, **1.17**, **1.18** to **1.24** of Ruina and Pratap's book.

    8. From Ruina and Pratap's book, solve the problems **1.1.1** to **1.3.16**.

    If you are new to scalars and vectors, you should solve these problems first by hand and then use Python to check the answers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - Read pages 44-92 of the first chapter of [Ruina and Pratap's book](http://ruina.tam.cornell.edu/Book/index.html) about scalars and vectors in mechanics.
    - [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb) — what the bases built here are for.
    - [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb) — position, displacement, velocity and acceleration as vectors.
    - [Matrix](https://github.com/BMClab/BMC/blob/master/notebooks/Matrix.ipynb) and [Rigid-body transformations in 3D](https://github.com/BMClab/BMC/blob/master/notebooks/Transformation3D.ipynb) — where these operations are assembled into rotations.

    ### Video lectures on the Internet

    - Khan Academy: [Vectors](https://www.khanacademy.org/math/algebra-home/alg-vectors)
    - [Vectors, what even are they?](https://youtu.be/fNk_zzaMoSs)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Ruina A, Pratap R (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
