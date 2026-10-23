# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App()




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rotating an ellipsoid about one axis

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil

    ## How to use this guide

    An ellipsoid makes orientation visible: unlike a sphere, its unequal axes let us see a rotation. Predict the long-axis direction before running each plot, then compare a rotated point with the rotated surface. The Exercises have collapsed worked solutions; the independent Problems provide final answers only.

    **Opening challenge.** The surface is $x^2+3y^2+15z^2=1$. Which axis is longest? If it is rotated about $x$, does that longest axis move?

    ## Python setup

    NumPy builds the surface mesh and Matplotlib renders it. All coordinates are embedded in this file. No external files are needed. You need matrix multiplication, sines and cosines.

    ## From the equation to a surface

    Write the ellipsoid as $(x/a)^2+(y/b)^2+(z/c)^2=1$. Its semiaxes are $a=1$, $b=1/\sqrt3$ and $c=1/\sqrt{15}$. The sampled angles describe surface points, not the orientation angles of the whole body. A rotation acts on the stacked Cartesian coordinates.

    **Exercise 1.** Find the three positive axis intercepts and check one in the original equation.

    /// details | Worked solution — Exercise 1

    The intercepts are $(1,0,0)$, $(0,1/\sqrt3,0)$ and $(0,0,1/\sqrt{15})$. For the second, $3(1/\sqrt3)^2=1$. The $x$ semiaxis is longest, and rotation about $x$ leaves it fixed.

    ///
    """)
    return

@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    # '%matplotlib notebook' command supported automatically in marimo
    plt.rcParams['figure.figsize']=10,10
    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1, 1, 1))

    coefs = (1, 3, 15)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1
    # Radii corresponding to the coefficients:
    rx, ry, rz = 1/np.sqrt(coefs)

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))


    xr = np.reshape(x, (1,-1))
    yr = np.reshape(y, (1,-1))
    zr = np.reshape(z, (1,-1))

    alpha = np.pi/3

    RX = np.array([[1,0,0],[0,np.cos(alpha),-np.sin(alpha)],[0,np.sin(alpha),np.cos(alpha)]])
    print(RX)
    rRotx = RX@np.vstack((xr,yr,zr))

    # Plot:
    ax.plot_surface(np.reshape(rRotx[0,:],(30,30)), np.reshape(rRotx[1,:],(30,30)),
                    np.reshape(rRotx[2,:],(30,30)), rstride=4, cstride=4, color='b')

    plt.xlabel('X')
    plt.ylabel('Y')
    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    plt.show()
    return (np,)


@app.cell
def _(np):
    mm = np.array([2.71, 10.22, 26.52])
    lm = np.array([2.92, 10.10, 18.85])
    fh = np.array([5.05, 41.90, 15.41])
    mc = np.array([8.29, 41.88, 26.52])
    ajc = (mm + lm)/2
    kjc = (fh + mc)/2
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)






@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rotated points and anatomical landmarks

    The final cell also retains four anatomical landmark coordinates from the original example. The ankle and knee centres are the midpoints of the malleoli and condyles. These are positions; the difference of the centres is a direction from which a segment basis can be built.

    **Exercise 2.** At $60^\circ$ about $x$, find the image of $(0,0,1)$ and check its length. Then calculate the ankle centre from the supplied malleoli.

    /// details | Worked solution — Exercise 2

    The last column of $R_x$ is $(0,-\sin60^\circ,\cos60^\circ)=(0,-\sqrt3/2,1/2)$, with squared length $3/4+1/4=1$. The ankle centre is $(2.815,10.160,22.685)$ cm. The knee centre is $(6.670,41.890,20.965)$ cm; keep the centimetre units when comparing with other datasets.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. What properties distinguish a rotation from scaling?
    2. Does transposing a rotation undo it?
    3. Are three orthogonal unit vectors automatically right-handed?

    /// details | Checkpoint answers

    1. A rotation preserves lengths and angles and has determinant +1. Scaling generally changes lengths. 2. Yes, for a proper orthogonal rotation matrix. 3. No; the determinant must also be +1.

    ///

    ## Problems

    1. Rotate $(0,1,0)$ by $60^\circ$ about the fixed positive $x$ axis.
    2. Give the ellipsoid’s volume and say whether rotation changes it.
    3. Let $R=R_z(90^\circ)R_x(90^\circ)$. Give $R(0,1,0)$ and compare it with the reversed order.

    /// details | Final answers to the problems

    1. $(0,1/2,\sqrt3/2)$.
    2. $4\pi/(3\sqrt{45})\approx0.6244$ cubic coordinate units; unchanged.
    3. $(0,0,1)$; the reversed order gives $(-1,0,0)$.

    ///

    ## Go deeper

    [Transformation3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py) develops active/passive conventions and sequence-dependent angles. [CoordinateSystem](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py) connects the marker calculations below the surfaces to anatomical frames.

    ### Video lectures on the Internet

    [Lynch and Park: rotation matrices](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-2-1-rotation-matrices-part-1-of-2/) explains the orthogonality and determinant tests used here.

    ## References

    - Lynch KM, Park FC, *Modern Robotics*, Chapter 3: rotation matrices and rigid configurations.

    - Original BMC ellipsoid and marker examples, preserved from the paired Jupyter notebook.
    """)
    return



if __name__ == "__main__":
    app.run()
