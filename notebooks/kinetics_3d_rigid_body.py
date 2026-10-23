# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Three-dimensional rigid body kinetics

    > Renato Naville Watanabe, Marcos Duarte<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    We will track angular momentum in laboratory and body frames, use principal inertias, and compare the equations with pendulum and posture data. Label the frame of every vector before taking a cross product or a derivative.

    Try the distributed **Exercises** before opening their collapsed worked solutions. Then use the **Checkpoint questions** and attempt the **Problems** independently; their panels contain final answers only. Predict the effect of a parameter change before running it, and check units and limiting cases.

    **Opening challenge.** A rigid body spins with a constant angular-velocity vector in its own frame. Can it still need an external torque?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    The dependency declaration at the top of this file lists the packages used by this guide. Marimo runs cells in dependency order. Run the notebook in an editor to use its displays and controls; the computational examples below explain their inputs before using them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Newton-Euler laws

    The Newton-Euler laws remain valid in the three-dimensional motion of a rigid body (for revision on Newton-Euler laws, [see this notebook about these laws](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/newton_euler_equations.py)).

    <span class="notranslate">$\vec{F} = m\vec{a_{cm}}$$\vec{M_O} = \frac{d\vec{H_O}}{dt}$</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Resultant force and moments

    The resultant force and the resultant moment around a point O are computed in the same way of the two-dimensional case.

    <span class="notranslate">$\vec{F} = \sum\limits_{i=1}^n \vec{F_n}$$\vec{M_O} = \sum\limits_{i=1}^n \vec{r_{i/O}} \times \vec{F_i}$</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deduction of the angular momentum in three-dimensional movement

    The major difference that appears when we are dealing with three-dimensional motions is to compute the derivative of the angular momentum. For the sake of simplicity, the analysis will only consider the point O as the center of mass of the body.

    We begin computing the angular momentum of a rigid body around its center of mass.

    <span class="notranslate">$\vec{H_{cm}} =  \int_B \vec{r_{/cm}} \times \vec{v}\,dm = \int  \vec{r_{/cm}} \times (\vec{\omega}\times\vec{r_{/cm}})\,dm$</span>
    where <span class="notranslate">$\vec{\boldsymbol{r_{/cm}}}$</span> is the vector from the point O to the position of the infinitesimal mass considered in the integral. For simplicity of the notation, we will use <span class="notranslate">$\vec{\boldsymbol{r}} = \vec{\boldsymbol{r_{/cm}}}$</span>. The triple vector product above can be computed using the rule:
        <span class="notranslate">$\vec{a}\times(\vec{b}\times\vec{c}) = (\vec{a}.\vec{c})\vec{b} - (\vec{a}.\vec{b})\vec{c}$.</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So, the angular momentum is:

    <span class="notranslate">$\vec{H_{cm}} = \int  \vec{r} \times (\vec{\omega}\times\vec{r})\,dm =
          \int  (r_x\hat{\boldsymbol{i}} + r_y\hat{\boldsymbol{j}}+r_z\hat{\boldsymbol{k}}) \times \left[(\omega_x\hat{\boldsymbol{i}} + \omega_y\hat{\boldsymbol{j}}+\omega_z\hat{\boldsymbol{k}})\times(r_x\hat{\boldsymbol{i}} + r_y\hat{\boldsymbol{j}}+r_z\hat{\boldsymbol{k}})\right]\,dm = \int \left[(r_x\hat{\boldsymbol{i}} + r_y\hat{\boldsymbol{j}}+r_z\hat{\boldsymbol{k}}).(r_x\hat{\boldsymbol{i}} + r_y\hat{\boldsymbol{j}}+r_z\hat{\boldsymbol{k}})\right](\omega_x\hat{\boldsymbol{i}} + \omega_y\hat{\boldsymbol{j}}+\omega_z\hat{\boldsymbol{k}}) -\left[(r_x\hat{\boldsymbol{i}} + r_y\hat{\boldsymbol{j}}+r_z\hat{\boldsymbol{k}}).(\omega_x\hat{\boldsymbol{i}} + \omega_y\hat{\boldsymbol{j}}+\omega_z\hat{\boldsymbol{k}})\right](r_x\hat{\boldsymbol{i}} + r_y\hat{\boldsymbol{j}}+r_z\hat{\boldsymbol{k}}) \,dm = \int (r_x^2 + r_y^2+r_z^2)(\omega_x\hat{\boldsymbol{i}} + \omega_y\hat{\boldsymbol{j}}+\omega_z\hat{\boldsymbol{k}}) - (r_x\omega_x + r_y\omega_y+r_z\omega_z)(r_x\hat{\boldsymbol{i}} + r_y\hat{\boldsymbol{j}}+r_z\hat{\boldsymbol{k}})\,dm$</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span class="notranslate">$\vec{H_{cm}} = \int (r_x^2 + r_y^2+r_z^2)\omega_x\hat{\boldsymbol{i}} - (r_x\omega_x + r_y\omega_y+r_z\omega_z)r_x\hat{\boldsymbol{i}} + (r_x^2 + r_y^2+r_z^2)\omega_y\hat{\boldsymbol{j}} - (r_x\omega_x + r_y\omega_y+r_z\omega_z)r_y\hat{\boldsymbol{j}}+(r_x^2 + r_y^2+r_z^2)\omega_z\hat{\boldsymbol{k}} - (r_x\omega_x + r_y\omega_y+r_z\omega_z)r_z\hat{\boldsymbol{k}}\,dm$</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span class="notranslate">$\vec{H_{cm}}=\left(\int \omega_x(r_y^2+r_z^2)\,dm+  \int-\omega_yr_xr_y\,dm + \int-\omega_z r_xr_z\,dm\right)\;\hat{\boldsymbol{i}} + \left(\int-\omega_x r_xr_y\,dm +\int\omega_y (r_x^2 +r_z^2)\,dm + \int- \omega_zr_yr_z\,dm\right)\hat{\boldsymbol{j}} + \left(\int-\omega_x r_xr_z\,dm  + \int-\omega_yr_yr_z\,dm + \int \omega_z(r_x^2+r_y^2) \,dm\right)\hat{\boldsymbol{k}}$</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span class="notranslate">$\vec{H_{cm}}=\bigg(\omega_x\int (r_y^2+r_z^2)\,dm+  \omega_y\int-r_xr_y\,dm + \omega_z\int- r_xr_z\,dm\bigg)\;\hat{\boldsymbol{i}}+\bigg(\omega_x\int- r_xr_y\,dm +\omega_y\int (r_x^2 +r_z^2)\,dm +  \omega_z\int-r_yr_z\,dm\bigg)\hat{\boldsymbol{j}} + \bigg(\omega_x\int- r_xr_z\,dm  + \omega_y\int-r_yr_z\,dm + \omega_z\int (r_x^2+r_y^2) \,dm\bigg)\hat{\boldsymbol{k}}$</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span class="notranslate">$\vec{H_{cm}}=\left(\omega_xI_{xx}^{cm}+  \omega_yI_{xy}^{cm} + \omega_zI_{xz}^{cm}\right)\;\hat{\boldsymbol{i}} + \left(\omega_xI_{xy}^{cm} +\omega_yI_{yy}^{cm} +  \omega_zI_{yz}^{cm}\right)\hat{\boldsymbol{j}}+\left(\omega_xI_{yz}^{cm}  + \omega_yI_{yz}^{cm} + \omega_zI_{zz}^{cm}\right)\hat{\boldsymbol{k}}$</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Angular momentum in three-dimensional movement


    \begin{align}
    \begin{split}
          \vec{H_{cm}}=\left[\begin{array}{ccc}I_{xx}^{cm}&I_{xy}^{cm}&I_{xz}^{cm}\\
                                      I_{xy}^{cm}&I_{yy}^{cm}&I_{yz}^{cm}\\
                                      I_{xz}^{cm}&I_{yz}^{cm}&I_{zz}^{cm}\end{array}\right]\cdot
                   \left[\begin{array}{c}\omega_x\\\omega_y\\\omega_z \end{array}\right] = I\vec{\omega}
          \end{split}
          \label{eq:angmom}
     \end{align}

     where this matrix is the Matrix of Inertia (or more rigorously, Tensor of Inertia) as defined previously in [this notebook about moment of inertia](center_of_mass_and_moment_of_inertia.py#matrix-of-inertia).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1 — Momentum need not follow velocity.** A body has principal inertia matrix $\operatorname{diag}(1,2,3)$ kg m² and body angular velocity $(1,1,0)$ rad/s. Find its angular momentum.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Solution to Exercise 1': mo.md(r"""
    $\mathbf H=\mathbf I\boldsymbol\omega=(1,2,0)$ kg m²/s. It is not parallel to $(1,1,0)$ because the first two principal inertias differ. Parallelism occurs along a principal axis or within a repeated-eigenvalue subspace.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The matrix of inertia is different depending on the body orientation

    So, to compute the angular momentum of a body we have to multiply the matrix of inertia $I$ of the body by its angular velocity $\vec{\omega}$. The problem on this approach is that the moments and products of inertia depends on the orientation of the body relative to the frame of reference. As they depend on the distances of each point of the body to the axes, if the body is rotating, the matrix of inertia $I$ will be different at each instant.

    <figure><img src="https://github.com/BMClab/BMC/blob/master/images/3Dbodyref.png?raw=1" width=800 />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The solution is to attach a frame reference to the body

    The solution to this problem is to attach a frame of reference to the body. We will denote this frame of reference as $\hat{\boldsymbol{e_1}}$, $\hat{\boldsymbol{e_2}}$ and $\hat{\boldsymbol{e_3}}$, with origin in the center of mass of the body. As can be noted from the figure below, the frame of reference moves along the body. Now the matrix of inertia $I$ will be constant relative to this new basis $\hat{\boldsymbol{e_1}}$, $\hat{\boldsymbol{e_2}}$ and $\hat{\boldsymbol{e_3}}$.

    <figure><img src="https://github.com/BMClab/BMC/blob/master/images/3DbodyrefMove.png?raw=1" width=800 />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The fixed frame is chosen in a way that the matrix of inertia is a diagonal matrix

    As we can choose the basis versors as we want, we can choose them so as to the products of inertia be equal to zero. This can always be done. If the body has axes of symmetry,  we can choose these axes (principal axes) as the direction of the basis to the products of inertia be equal to zero. Having the basis $\hat{\boldsymbol{e_1}}$, $\hat{\boldsymbol{e_2}}$ and $\hat{\boldsymbol{e_3}}$, the matrix of inertia will be a diagonal matrix:

    $$
    I = \left[\begin{array}{ccc}I_1&0&0\\
              0&I_2&0\\
              0&0&I_3\end{array}\right]
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The angular velocity in the fixed frame

    So, we can write the angular momentum vector relative to this new basis. To do this, we must write the angular velocity in this new basis:

    <span class="notranslate">$\vec{\omega} = \omega_1\hat{\boldsymbol{e_1}} + \omega_2\hat{\boldsymbol{e_2}} + \omega_3\hat{\boldsymbol{e_3}}$</span>

    Note that this angular velocity is the same vector that we used previously. We are just describing it in a basis attached to the body (local basis).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The angular momentum in the fixed frame

    So, using the basis in the direction of the principal axes of the body the angular momentum simplifies to:

    <span class="notranslate">$\vec{H_{cm}} = I\vec{\omega} = I_1\omega_1 \hat{\boldsymbol{e_1}} + I_2\omega_2 \hat{\boldsymbol{e_2}} +I_3\omega_3 \hat{\boldsymbol{e_3}}$</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2 — Differentiate a rotating basis.** Use the same inertia and angular velocity with zero body-frame angular acceleration. Find the required body-frame torque.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Solution to Exercise 2': mo.md(r"""
    The transport theorem gives $\mathbf M=\mathbf I\dot{\boldsymbol\omega}+\boldsymbol\omega\times(\mathbf I\boldsymbol\omega)$. Here the first term vanishes and $(1,1,0)\times(1,2,0)=(0,0,1)$ N m. This resolves the opening challenge.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Derivative of the angular momentum

    For the second Newton-Euler law, we must compute the derivative of the angular momentum. So, we derive the angular momentum in Eq. \eqref{eq:angmomprinc}. As the versors $\hat{\boldsymbol{e_1}}$, $\hat{\boldsymbol{e_2}}$ and $\hat{\boldsymbol{e_3}}$ are varying in time, we must consider their derivatives.

    $$
    \frac{d\vec{H_{cm}}}{dt} = I_1\dot{\omega_1}\hat{\boldsymbol{e_1}} + I_2\dot{\omega_2}\hat{\boldsymbol{e_2}}+I_3\dot{\omega_3}\hat{\boldsymbol{e_3}} + I_1\omega_1\frac{d\hat{\boldsymbol{e_1}}}{dt} + I_2\omega_2\frac{d\hat{\boldsymbol{e_2}}}{dt}+I_3\omega_3\frac{d\hat{\boldsymbol{e_3}}}{dt}
    $$

    Now it only remains to find an expression for the derivative of the versors $\frac{d\hat{\boldsymbol{e_1}}}{dt}$, $\frac{d\hat{\boldsymbol{e_2}}}{dt}$ and <span class="notranslate">$\frac{d\hat{\boldsymbol{e_3}}}{dt}$<span>.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Angular velocity


    The angular velocity can be written as ([here you can learn more about the computation of the angular velocity](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/angular_velocity_3d.py)):

    <span class="notranslate">$\vec{\omega} =  \left(\frac{d\hat{\boldsymbol{e_2}}}{dt}\cdot \hat{\boldsymbol{e_3}}\right) \hat{\boldsymbol{e_1}} + \left(\frac{d\hat{\boldsymbol{e_3}}}{dt}\cdot \hat{\boldsymbol{e_1}}\right)
        \hat{\boldsymbol{e_2}} + \left(\frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot \hat{\boldsymbol{e_2}}\right) \hat{\boldsymbol{e_3}}$</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The derivative of the versors

    Now, we must isolate the derivative of the versors to substitute them in the Eq. \eqref{eq:derivangmom}. To isolate the derivative of the versor $\hat{\boldsymbol{e_1}}$, first we cross multiply both sides of the equation above by $\hat{\boldsymbol{e_1}}$:

    $$
    \vec{\omega} \times \hat{\boldsymbol{e_1}} = - \left(\frac{d\hat{\boldsymbol{e_3}}}{dt}\cdot \hat{\boldsymbol{e_1}}\right) \hat{\boldsymbol{e_3}} + \left(\frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot \hat{\boldsymbol{e_2}}\right) \hat{\boldsymbol{e_2}}
    $$

    If we note that the term multipliying $\hat{\boldsymbol{e_3}}$ in the right side of the identity can be obtained by $\frac{d\hat{\boldsymbol{e_1}}\cdot\hat{\boldsymbol{e_3}} }{dt} = \frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot\hat{\boldsymbol{e_3}} + \frac{d\hat{\boldsymbol{e_3}}}{dt}\cdot\hat{\boldsymbol{e_1}} \rightarrow 0 = \frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot\hat{\boldsymbol{e_3}} + \frac{d\hat{\boldsymbol{e_3}}}{dt}\cdot\hat{\boldsymbol{e_1}} \rightarrow \frac{d\hat{\boldsymbol{e_3}}}{dt}\cdot\hat{\boldsymbol{e_1}}  = - \frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot\hat{\boldsymbol{e_3}}$ (the scalar product $\hat{\boldsymbol{e_1}}\cdot\hat{\boldsymbol{e_3}}$ is zero because these vectors are orthogonal), we can write the equation above becomes:

    $$
    \vec{\omega} \times \hat{\boldsymbol{e_1}} = \left(\frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot \hat{\boldsymbol{e_3}}\right) \hat{\boldsymbol{e_3}} + \left(\frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot \hat{\boldsymbol{e_2}}\right) \hat{\boldsymbol{e_2}}
    $$

    Finally, we can note that $\frac{d\hat{\boldsymbol{e_1}}\cdot\hat{\boldsymbol{e_1}} }{dt} = \frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot\hat{\boldsymbol{e_1}} + \frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot\hat{\boldsymbol{e_1}} \rightarrow \frac{d(1)}{dt} = 2\frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot\hat{\boldsymbol{e_1}} \rightarrow \frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot\hat{\boldsymbol{e_1}}  = 0$. As this term is equal to zero, we can add it to the expression above:

    $$
    \vec{\boldsymbol{\omega}} \times \hat{\boldsymbol{e_1}} = \left(\frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot\hat{\boldsymbol{e_1}}\right)\hat{\boldsymbol{e_1}} +  \left(\frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot \hat{\boldsymbol{e_3}}\right) \hat{\boldsymbol{e_3}} + \left(\frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot \hat{\boldsymbol{e_2}}\right) \hat{\boldsymbol{e_2}}
    $$

    Note that the expression above is just another manner to write the vector $\frac{d\hat{\boldsymbol{e_1}}}{dt}$, as any vector can be described by the sum of the projections on each of the versors forming a basis.

    So, the derivative of the versor $\hat{\boldsymbol{e_1}}$  can be written as:

    $$
    \frac{d\hat{\boldsymbol{e_1}}}{dt} = \vec{\boldsymbol{\omega}} \times \hat{\boldsymbol{e_1}}
    $$

    Similarly, the derivatives of the versors $\hat{\boldsymbol{e_2}}$ and $\hat{\boldsymbol{e_3}}$ can be written as:

    $$
    \frac{d\hat{\boldsymbol{e_2}}}{dt} = \vec{\boldsymbol{\omega}} \times \hat{\boldsymbol{e_2}} ~~~~~~~~\text{and} ~~~~~~ \frac{d\hat{\boldsymbol{e_3}}}{dt} = \vec{\boldsymbol{\omega}} \times \hat{\boldsymbol{e_3}}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The derivative of the angular momentum

    Now we can get back to the equation describing the derivative of the angular momentum:

    <span class="notranslate">
    \begin{align}
        \begin{split}
        \frac{d\vec{H_{cm}}}{dt} =&I_1\dot{\omega_1}\hat{\boldsymbol{e_1}} + I_2\dot{\omega_2}\hat{\boldsymbol{e_2}}+I_3\dot{\omega_3}\hat{\boldsymbol{e_3}} + I_1\omega_1\frac{d\hat{\boldsymbol{e_1}}}{dt} + I_2\omega_2\frac{d\hat{\boldsymbol{e_2}}}{dt}+I_3\omega_3\frac{d\hat{\boldsymbol{e_3}}}{dt}=\\
        =& I_1\dot{\omega_1}\hat{\boldsymbol{e_1}} + I_2\dot{\omega_2}\hat{\boldsymbol{e_2}}+I_3\dot{\omega_3}\hat{\boldsymbol{e_3}} + I_1\omega_1(\vec{\boldsymbol{\omega}} \times \hat{\boldsymbol{e_1}}) + I_2\omega_2(\vec{\boldsymbol{\omega}} \times \hat{\boldsymbol{e_2}})+I_3\omega_3(\vec{\boldsymbol{\omega}} \times \hat{\boldsymbol{e_3}}) = \\
       =& I_1\dot{\omega_1}\hat{\boldsymbol{e_1}} + I_2\dot{\omega_2}\hat{\boldsymbol{e_2}}+I_3\dot{\omega_3}\hat{\boldsymbol{e_3}} + \vec{\boldsymbol{\omega}} \times I_1\omega_1\hat{\boldsymbol{e_1}} + \vec{\boldsymbol{\omega}} \times I_2\omega_2\hat{\boldsymbol{e_2}}+\vec{\boldsymbol{\omega}} \times I_3\omega_3\hat{\boldsymbol{e_3}} = \\
       =& I_1\dot{\omega_1}\hat{\boldsymbol{e_1}} + I_2\dot{\omega_2}\hat{\boldsymbol{e_2}}+I_3\dot{\omega_3}\hat{\boldsymbol{e_3}} + \vec{\boldsymbol{\omega}} \times (I_1\omega_1\hat{\boldsymbol{e_1}} + I_2\omega_2\hat{\boldsymbol{e_2}} +  I_3\omega_3\hat{\boldsymbol{e_3}})=\\
       =&I\vec{\dot{\omega}} + \vec{\omega} \times (I\vec{\omega})
       \end{split}
       \label{eq:derivangmomVec}
    \end{align}
    </span>

    Performing the cross products, we can get the expressions for each of the coordinates attached to the body:

    <span class="notranslate">
    \begin{align}
        \begin{split}
        \frac{d\vec{H_{cm}}}{dt} =\left[\begin{array}{c}I_1\dot{\omega_1}\\I_2\dot{\omega_2}\\I_3\dot{\omega_3}\end{array}\right] + \left[\begin{array}{c}\omega_1\\\omega_2\\\omega_3\end{array}\right]  \times \left[\begin{array}{c}I_1\omega_1\\I_2\omega_2\\I_3\omega_3\end{array}\right]  = \left[\begin{array}{c}I_1\dot{\omega_1} + \omega_2\omega_3(I_3-I_2)\\I_2\dot{\omega_2}+\omega_1\omega_3(I_1-I_3)\\I_3\dot{\omega_3}+\omega_1\omega_2(I_2-I_1)\end{array}\right]
       \end{split}
    \end{align}
    </span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Newton-Euler laws

    Having computed the derivative of the angular momentum, we have the final forms of the Newton-Euler laws:

    <span class="notranslate">$F_x = ma_{cm_x}$$F_y = ma_{cm_y}$$F_z = ma_{cm_z}$$M_{cm_1} = I_1\dot{\omega_1} + \omega_2\omega_3(I_3-I_2)$$M_{cm_2} = I_2\dot{\omega_2}+\omega_1\omega_3(I_1-I_3)$$M_{cm_3} = I_3\dot{\omega_3}+\omega_1\omega_2(I_2-I_1)$</span>

    Or, in the vectorial form:

    <span class="notranslate">$\vec{F} = m\vec{a_{cm}}$$\vec{M_{cm}} = I\dot{\vec{\omega}}+\vec{\omega}\times(I\vec{\omega}) \text{            (vectors in local frame)}$</span>


    Note that the equations of the forces are written in the global frame of reference and the equations of the moments are described in the frame of reference of the body. So, before using Eq.~\eqref{eq:derivangmomVec} or the equations Eq.\eqref{eq:M1},\eqref{eq:M2} and \eqref{eq:M3} you must transform all the forces and moment-arms to the frame of reference of the body by using rotation matrices.

    Below are shown some examples with three-dimensional kinematic data to find the forces and moments acting on the body.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3 — Rotate the inertia tensor.** A body has $I_B=\operatorname{diag}(1,2,3)$ kg m². A 90° rotation about z maps body components to laboratory components. What is $I_G$?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Solution to Exercise 3': mo.md(r"""
    With $\mathbf v_G=R\mathbf v_B$, the tensor is $I_G=RI_BR^T=\operatorname{diag}(2,1,3)$ kg m². Both sides of the tensor transform; multiplying on only one side is not a change of basis.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Examples
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###  3D pendulum bar

     At the file '../data/3Dpendulum.txt' there are 3 seconds of data of 3 points of the three-dimensional cylindrical pendulum, very similar to the pendulum shown in the [notebook about free-body diagrams](free_body_diagram_for_rigid_bodies.py#pendulum), except that it can move in every direction. Also it has a motor at the upper part of the cylindrical bar producing torques to move the bar. It has mass $m=1$ kg, length $l=1$ m and radius $r=0.1$ m.

     The point m1 is at the upper part of the cylinder and is the origin of the system.

     The point m2 is at the center of mass of the cylinder.

     The point m3 is a point at the surface of the cylinder.

    The free-body diagram of the 3d pendulum is depicted below. There is the gravitational force acting at the center of mass of gravity of the body and the torque $\vec{M_1}$ due to the motor acting on the pendulum and the force $\vec{F_1}$ due to the restraint at the upper part of the cylinder. Together with the forces, the local basis $\hat{\boldsymbol{e_1}}$, $\hat{\boldsymbol{e_2}}$ and $\hat{\boldsymbol{e_3}}$ in the direction of the principal axes an origin at the center of mass of the body is also shown.

    <figure><img src="https://github.com/BMClab/BMC/blob/master/images/3DpendulumFBD.png?raw=1" width=400 />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The resultant forces acting on the cylinder is:

    <span class="notranslate">$\vec{F} = \vec{F_O} - mg\hat{\boldsymbol{k}}$</span>

    So, the first Newton-Euler law, at each component of the global  basis, is written as:

    <span class="notranslate">
    \begin{align}
        \begin{split}
            F_{O_x} &=  ma_{cm_x} &\rightarrow  F_{O_x} &=  ma_{cm_x} \\
            F_{O_y} &= ma_{cm_y} &\rightarrow  F_{O_y} &=  ma_{cm_y}\\
            F_{O_z} - mg &= ma_{cm_z} &\rightarrow  F_{O_z} &=  ma_{cm_z} + mg
        \end{split}
        \label{eq:fnependulum}
    \end{align}
    </span >
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, the resultant moment applied to the body, computed relative to the center of mass, is:

    $$
    \vec{M} = \vec{M_O} + \vec{r_{O/cm}} \times \vec{F_O}
    $$

    So, the second Newton-Euler law, at each of the components at the local basis of the body, is written as:

    $$
    \begin{aligned}
    \begin{split}
            M_{O_1} + MFocm_1 &= I_1\dot{\omega_1} + \omega_2\omega_3(I_3-I_2) \rightarrow M_{O_1} &= I_1\dot{\omega_1} + \omega_2\omega_3(I_3-I_2) -  MFocm_1\\
            M_{O_2} + MFocm_2 &= I_2\dot{\omega_2} + \omega_1\omega_3(I_1-I_3) \rightarrow M_{O_2} &= I_2\dot{\omega_2} + \omega_1\omega_3(I_1-I_3) -  MFocm_2\\
            M_{O_3} + MFocm_3 &= I_3\dot{\omega_3} + \omega_1\omega_2(I_2-I_1) \rightarrow M_{O_3} &= I_3\dot{\omega_3} + \omega_1\omega_2(I_2-I_1) -  MFocm_3
        \end{split}
    \end{aligned}
    $$

    <span class="notranslate">

    where <span class="notranslate">$\vec{MFocm} = \vec{r_{O/cm}} \times \vec{F_O}$.

    The moments of inertia at the directions of $\hat{\boldsymbol{e_1}}$, $\hat{\boldsymbol{e_2}}$ and $\hat{\boldsymbol{e_3}}$ are, $I_1 = \frac{mR^2}{12}$ and $I_2=I_3=\frac{m(3R^2+l^2)}{12}$. Now, to compute the moment $\vec{M_O}$ and the force  $\vec{F_O}$, we will need to find the acceleration of the center of mass $\vec{a_{cm}}$, the angular velocity  $\vec{\omega}$, the time-derivatives of each component of the angular velocity, and the moment-arm $\vec{r_{O/cm}}$ to compute the torque due to the force $\vec{F_O}$. These signals will come from the kinematic data file.

    First,  we need to open the file with the data:
    """)
    return


@app.function
def repo_file(path, *, base_url="https://raw.githubusercontent.com/BMClab/BMC/master/"):
    """Find a repository file beside this notebook, or fetch a temporary copy.

    Local files take precedence. Only file acquisition falls back to GitHub;
    parsing errors remain visible to the calling notebook. urllib3 selects its
    browser transport automatically without platform-specific imports.
    """
    import hashlib
    import tempfile
    from pathlib import Path, PurePosixPath
    from urllib.parse import quote

    import marimo as mo
    import urllib3

    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(path):
        raise ValueError("Use a repository-relative path such as data/walk.trc")
    location = mo.notebook_dir()
    if location is not None:
        directory = Path(location).resolve()
        for base in (directory, *directory.parents):
            local = base.joinpath(*relative.parts)
            if local.is_file():
                return local
    url = base_url.rstrip("/") + "/" + quote(relative.as_posix(), safe="/")
    key = hashlib.sha256(base_url.rstrip("/").encode()).hexdigest()[:16]
    cached = (Path(tempfile.gettempdir()) / "bmc-notebooks-v2" / key).joinpath(*relative.parts)
    if not cached.is_file():
        response = urllib3.request("GET", url, timeout=30.0, retries=2)
        if response.status != 200:
            raise OSError(f"Cannot load {path}: HTTP {response.status} from {url}")
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as download:
            temporary = Path(download.name)
            download.write(response.data)
        try:
            temporary.replace(cached)
        finally:
            temporary.unlink(missing_ok=True)
    return cached


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=2)
    f = open(repo_file('data/3dPendulum.txt'))
    data = np.loadtxt(repo_file('data/3dPendulum.txt'), 
                      skiprows=1, delimiter = ',')
    header = f.readline()
    print(header)
    print(data)
    return data, np, plt


@app.cell
def _():
    m = 1
    g = 9.81
    l = 1
    _r = 0.1
    I1 = m * _r ** 2 / 12
    I2 = m * (3 * _r ** 2 + l ** 2) / 12
    I3 = I2
    return I1, I2, I3, g, m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we assign the proper columns to variables:
    """)
    return


@app.cell
def _(data):
    t = data[:,0]
    m1 = data[:,1:4]
    m2 = data[:,4:7]
    m3 = data[:,7:]
    return m1, m2, m3, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As the center of mass is the data contained in m2, we can find the acceleration of the center of mass $\vec{\boldsymbol{a_{cm}}}$. This will be performed by deriving numerically the position of the center of mass twice. The numerical derivative of a function $f(t)$ with the samples $f(i)$ can be performed by taking the forward difference of the values $f(i)$:

    $$
    \frac{df}{dt}(i) = \frac{f(i+1)-f(i)}{\Delta t}
    $$

    The numerical derivative could be obtained by taking the backward differences as well:

    $$
    \frac{df}{dt}(i) = \frac{f(i)-f(i-1)}{\Delta t}
    $$

    A better estimation of the derivative of the time derivative of the function $f(t)$ would be obtained by the average value between the estimations using the backward and forward differences (this subject is treated [in this notebook about data filtering](data_filtering.py#kinematics-of-a-ball-toss)):

    $$
    \frac{df}{dt}(i) = \frac{\frac{f(i+1)-f(i)}{\Delta t} + \frac{f(i)-f(i-1)}{\Delta t}}{2} = \frac{f(i+1)-f(i-1)}{2\Delta t}
    $$

    So, the acceleration of the center of mass, is:
    """)
    return


@app.cell
def _(m2, t):
    dt = t[1] - t[0]
    rcm = m2
    _vcm = (rcm[2:, :] - rcm[0:-2, :]) / (2 * dt)
    acm = (_vcm[2:, :] - _vcm[0:-2, :]) / (2 * dt)
    print(rcm.shape)
    print(acm.shape)
    return acm, dt, rcm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can find the force <span class="notranslate">$\vec{\boldsymbol{F_O}}$</span>  using the Eq. \eqref{eq:fnependulum}.
    """)
    return


@app.cell
def _(acm, g, m, np, plt, t):
    Fox = m*acm[:,0]
    Foy = m*acm[:,1]
    Foz = m*acm[:,2] + m*g
    Fo=np.hstack((Fox.reshape(-1,1),Foy.reshape(-1,1),Foz.reshape(-1,1)))

    plt.figure()
    plt.plot(t[0:acm.shape[0]], Fox)
    plt.plot(t[0:acm.shape[0]], Foy, '--')
    plt.plot(t[0:acm.shape[0]], Foz)
    plt.legend(('x','y','z'))
    plt.title('Force (N)')
    plt.show()
    return (Fo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, to find the moment being applied to the body, we need to compute a basis attached to the body
    """)
    return


@app.cell
def _(m1, m2, m3, np):
    e1 = m2 - m1
    e1 = e1/np.linalg.norm(e1,axis=1,keepdims=True)

    e2_temp = m3-m2
    e2_temp = e2_temp/np.linalg.norm(e2_temp,axis=1,keepdims=True)

    e3 = np.cross(e1,e2_temp,axis=1)
    e3 = e3/np.linalg.norm(e3,axis=1,keepdims=True)

    e2 = np.cross(e3,e1, axis=1)
    e2 = e2/np.linalg.norm(e2,axis=1,keepdims=True)
    return e1, e2, e3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To compute the moment applied to the body, we need the angular velocity described in the basis attached to the body. The easiest way to find this angular velocity is to use Eq. \eqref{eq:angvel}, repeated here.

    <span class="notranslate">$\vec{\omega} =  \left(\frac{d\hat{\boldsymbol{e_2}}}{dt}\cdot \hat{\boldsymbol{e_3}}\right) \hat{\boldsymbol{e_1}} + \left(\frac{d\hat{\boldsymbol{e_3}}}{dt}\cdot \hat{\boldsymbol{e_1}}\right) \hat{\boldsymbol{e_2}} + \left(\frac{d\hat{\boldsymbol{e_1}}}{dt}\cdot \hat{\boldsymbol{e_2}}\right) \hat{\boldsymbol{e_3}}$</span>


     To do this we need the derivatives of the basis versors. This will also be performed with Eq. \eqref{eq:centralderiv}.

     To perform the computation of the angular velocity remember that the scalar product between two vectors is given by:

     <span class="notranslate">$\vec{v}\cdot\vec{w} = \left[\begin{array}{c}v_x\\v_y\\v_z \end{array}\right]\cdot \left[\begin{array}{c}w_x\\w_y\\w_z \end{array}\right] =  v_x.w_x+v_yw_y+v_zw_z$</span>
    """)
    return


@app.cell
def _(dt, e1, e2, e3, np):
    _de1dt = (e1[2:, :] - e1[0:-2, :]) / (2 * dt)
    _de2dt = (e2[2:, :] - e2[0:-2, :]) / (2 * dt)
    _de3dt = (e3[2:, :] - e3[0:-2, :]) / (2 * dt)
    omega = np.hstack((np.sum(_de2dt * e3[1:-1, :], axis=1, keepdims=True), np.sum(_de3dt * e1[1:-1, :], axis=1, keepdims=True), np.sum(_de1dt * e2[1:-1, :], axis=1, keepdims=True)))
    return (omega,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From the angular velocity vector we can obtain the derivatives of each component of it, also needed to compute the moment applied to the body. To do this we will use Eq. \eqref{eq:centralderiv}:
    """)
    return


@app.cell
def _(dt, omega):
    alpha = (omega[2:,:]-omega[0:-2,:])/(2*dt)
    return (alpha,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It remains to find the moment caused by the force $\vec{F_O}$, $\vec{MFocm} = \vec{r_{O/cm}} \times \vec{F_O}$. The moment-arm <span class="notranslate">$\vec{r_{O/cm}} =-\vec{r_{cm}}$.
    """)
    return


@app.cell
def _(Fo, np, rcm):
    MFocm = np.cross(-rcm[2:-2], Fo)
    return (MFocm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The problem is that this moment is in the global basis. We need to transform it to the local basis. This will be performed using the rotation matrix of the bar. Each row of this matrix is one of the basis versors. Note that at each instant the matrix of rotation $R$ will be different.  After the matrix is formed, we can find the components of the moment $\vec{MFocm}$ in the local basis by multiplying the matrix of rotation $R$ by the vector $\vec{MFocm}$.
    """)
    return


@app.cell
def _(MFocm, e1, e2, e3, np):
    MFocmLocal = np.zeros_like(MFocm)
    for _i in range(MFocm.shape[0]):
        _R = np.vstack((e1[_i, :], e2[_i, :], e3[_i, :]))
        MFocmLocal[_i, :] = _R @ MFocm[_i, :]
    return (MFocmLocal,)


@app.cell
def _(I1, I2, I3, MFocmLocal, alpha, omega, plt, t):
    Mo1 = I1*alpha[:,0] + omega[0:alpha.shape[0],1]*omega[0:alpha.shape[0],2]*(I3-I2) - MFocmLocal[:,0]
    Mo2 = I2*alpha[:,1] + omega[0:alpha.shape[0],0]*omega[0:alpha.shape[0],2]*(I1-I3) - MFocmLocal[:,1]
    Mo3 = I3*alpha[:,2] + omega[0:alpha.shape[0],0]*omega[0:alpha.shape[0],1]*(I2-I1) - MFocmLocal[:,2]
    plt.figure()
    plt.plot(t[2:-2], Mo1)
    plt.plot(t[2:-2], Mo2)
    plt.plot(t[2:-2], Mo3)
    plt.legend(('$e_1$','$e_2$','$e_3$'))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could also have used the vectorial form of the derivative of the angular momentum (Eq. \eqref{eq:derivangmomVec}) and instead of writing three lines of code, write only one. The result is the same.
    """)
    return


@app.cell
def _(I1, I2, I3, MFocmLocal, alpha, np, omega, plt, t):
    _I = np.array([[I1, 0, 0], [0, I2, 0], [0, 0, I3]])
    Mo = (_I @ alpha.T).T + np.cross(omega[0:alpha.shape[0], :], (_I @ omega[0:alpha.shape[0], :].T).T, axis=1) - MFocmLocal
    plt.figure()
    plt.plot(t[2:-2], Mo)
    plt.legend(('$e_1$', '$e_2$', '$e_3$'))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4 — Choose a derivative strategy.** Why can angular acceleration computed from marker data be much noisier than angular velocity, even with an exact differentiation formula?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Solution to Exercise 4': mo.md(r"""
    Differentiation amplifies high-frequency measurement noise. A second derivative amplifies it more strongly. Inspect sampling, coordinate continuity and filtering before differentiating, then report the filter and assess sensitivity rather than interpreting every spike as a physical torque.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data from postural control

    This example will use real data from a subject during quiet standing during 60 seconds. This data is from the database freely available at [https://github.com/demotu/datasets/tree/master/PDS](https://github.com/demotu/datasets/tree/master/PDS). The data of this subject is in the file '../data/postureData.txt'.

    The mass of the subject was $m = 53$ kg and her height was $h= 1.65$ m.

       The free-body diagram is very similar to the free-body diagram shown [in the notebook about free-body diagram](free_body_diagram_for_rigid_bodies.py#human-quiet-standing), except that the force $\vec{F_A}$ and the moment $\vec{M_A}$ have components at all 3 directions.

       So, the first Newton-Euler law, at each component of the global  basis, is written as (note that in these data, the vertical direction is the y coordinate):

    $$
    \begin{aligned}
    \begin{split}
            F_{A_x} &=  ma_{cm_x} &\rightarrow  F_{A_x} &=  ma_{cm_x} \\
            F_{A_y}  - mg &= ma_{cm_y} &\rightarrow  F_{A_y} &=  ma_{cm_y} + mg\\
            F_{A_z} &= ma_{cm_z} &\rightarrow  F_{A_z} &=  ma_{cm_z}
        \end{split}
    \end{aligned}
    $$

    Now, the resultant moment applied to the body, computed relative to the center of mass, is:

    $$
    \vec{M} = \vec{M_A} + \vec{r_{A/cm}} \times \vec{F_A}
    $$

    So, the second Newton-Euler law, at each of the components at the local basis of the body, is written as:

    $$
    \begin{aligned}
    \begin{split}
            \vec{M_A} + \vec{MFacm} &= I\left[\begin{array}{c}\dot{\omega_1}\\\dot{\omega_2}\\\dot{\omega_3}\end{array}\right] + \vec{\omega}\times  I\vec{\omega} \rightarrow \vec{M_A} = I\left[\begin{array}{c}\dot{\omega_1}\\\dot{\omega_2}\\\dot{\omega_3}\end{array}\right] + \vec{\omega} \times  I\vec{\omega}- \vec{MFacm}
        \end{split}
    \end{aligned}
    $$

    where  $\vec{MFAcm} = \vec{r_{A/cm}} \times \vec{F_A}$.

    Now we open the data and assign the coordinates of each marker to a variable.
    """)
    return


@app.cell
def _(np):
    data_1 = np.loadtxt(repo_file('data/postureData.txt'), skiprows=1, delimiter=',')
    t_1 = data_1[:, 0]
    dt_1 = t_1[2] - t_1[1]
    rcm_1 = data_1[:, 1:4]
    rrA = data_1[:, 4:7]
    rlA = data_1[:, 7:]
    return dt_1, rcm_1, rlA, rrA, t_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The body will be approximated by a cylinder with the height of the subject and radius equal to half of the mean distances between the right and left medial malleoli.
    """)
    return


@app.cell
def _(np, rlA, rrA):
    m_1 = 53
    h = 1.65
    _r = np.mean(np.linalg.norm(rrA - rlA, axis=1)) / 2
    I1_1 = m_1 * _r ** 2 / 12
    I2_1 = m_1 * (3 * _r ** 2 + h ** 2) / 12
    I3_1 = I2_1
    return I1_1, I2_1, I3_1, m_1


@app.cell
def _(dt_1, g, m_1, np, rcm_1):
    _vcm = (rcm_1[2:, :] - rcm_1[0:-2, :]) / (2 * dt_1)
    acm_1 = (_vcm[2:, :] - _vcm[0:-2, :]) / (2 * dt_1)
    FAx = m_1 * acm_1[:, 0]
    FAy = m_1 * acm_1[:, 1] + m_1 * g
    FAz = m_1 * acm_1[:, 2]
    FA = np.hstack((FAx.reshape(-1, 1), FAy.reshape(-1, 1), FAz.reshape(-1, 1)))
    return (FA,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we form the basis attached to the body. The first versor $\hat{\boldsymbol{e_1}}$ will be a versor from the midpoint between the medial malleoli and the center of mass of the body. The second versor $\hat{\boldsymbol{e_2}}$ will be a versor from  the right to the left malleolus. The third versor $\hat{\boldsymbol{e_3}}$ will be a cross product between $\hat{\boldsymbol{e_1}}$ and $\hat{\boldsymbol{e_2}}$.
    """)
    return


@app.cell
def _(np, rcm_1, rlA, rrA):
    e1_1 = rcm_1 - (rlA + rrA) / 2
    e1_1 = e1_1 / np.linalg.norm(e1_1, axis=1, keepdims=True)
    e2_1 = rlA - rrA
    e2_1 = e2_1 / np.linalg.norm(e2_1, axis=1, keepdims=True)
    e3_1 = np.cross(e1_1, e2_1, axis=1)
    e3_1 = e3_1 / np.linalg.norm(e3_1, axis=1, keepdims=True)
    e2_1 = np.cross(e3_1, e1_1, axis=1)
    e2_1 = e2_1 / np.linalg.norm(e2_1, axis=1, keepdims=True)
    return e1_1, e2_1, e3_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can find the angular velocity <span class="notranslate">$\vec{\omega}$</span> at the basis attached to the body using Eq.\eqref{eq:angvel} and the time derivatives of its components.
    """)
    return


@app.cell
def _(dt_1, e1_1, e2_1, e3_1, np):
    _de1dt = (e1_1[2:, :] - e1_1[0:-2, :]) / (2 * dt_1)
    _de2dt = (e2_1[2:, :] - e2_1[0:-2, :]) / (2 * dt_1)
    _de3dt = (e3_1[2:, :] - e3_1[0:-2, :]) / (2 * dt_1)
    omega_1 = np.hstack((np.sum(_de2dt * e3_1[1:-1, :], axis=1).reshape(-1, 1), np.sum(_de3dt * e1_1[1:-1, :], axis=1).reshape(-1, 1), np.sum(_de1dt * e2_1[1:-1, :], axis=1).reshape(-1, 1)))
    alpha_1 = (omega_1[2:, :] - omega_1[0:-2, :]) / (2 * dt_1)
    return alpha_1, omega_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we need to find the moment caused by the force at the ankles $\vec{F_A}$, $\vec{MFAcm} = \vec{r_{A/cm}} \times \vec{F_A}$. The moment-arm $\vec{r_{A/cm}}$ is the vector from the center of mass to the midpoint of the lateral malleoli.

    Besides the description of the moment due to the force $\vec{F_A}$ in the basis attached to the body, we will also describe the force $\vec{F_A}$ at the local basis. This is useful because it has an anatomical meaning. After this we can use the equations of Newton-Euler to obtain the moment at the ankle.

    After having all signals described in the basis of the body, the moment being applied by the muscles of the ankle is computed using Eq. \eqref{eq:derivangmomVec}.
    """)
    return


@app.cell
def _(
    FA,
    I1_1,
    I2_1,
    I3_1,
    alpha_1,
    e1_1,
    e2_1,
    e3_1,
    np,
    omega_1,
    plt,
    rcm_1,
    rlA,
    rrA,
    t_1,
):
    racm = (rlA + rrA) / 2 - rcm_1
    MFAcm = np.cross(racm[0:FA.shape[0], :], FA)
    MFAcmLocal = np.zeros_like(MFAcm)
    FALocal = np.zeros_like(MFAcm)
    for _i in range(MFAcm.shape[0]):
        _R = np.vstack((e1_1[_i, :], e2_1[_i, :], e3_1[_i, :]))
        MFAcmLocal[_i, :] = _R @ MFAcm[_i, :]
        FALocal[_i, :] = _R @ FA[_i, :]
    _I = np.diag([I1_1, I2_1, I3_1])
    MA = (_I @ alpha_1.T).T + np.cross(omega_1[0:alpha_1.shape[0], :], (_I @ omega_1[0:alpha_1.shape[0], :].T).T, axis=1) - MFAcmLocal
    plt.figure()
    plt.plot(t_1[2:-2], MA)
    plt.legend(('longitudinal', 'sagittal', 'mediolateral'))
    plt.title('Ankle Torque')
    plt.xlabel('t (s)')
    plt.ylabel('M (N.m)')
    plt.show()
    plt.figure()
    plt.plot(t_1[2:-2], FALocal[:, 0])
    plt.plot(t_1[2:-2], FALocal[:, 1])
    plt.plot(t_1[2:-2], FALocal[:, 2])
    plt.title('Ankle Force')
    plt.legend(('longitudinal', 'sagittal', 'mediolateral'))
    plt.xlabel('t (s)')
    plt.ylabel('F (N)')
    plt.show()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. When is $\mathbf H$ parallel to $\boldsymbol\omega$?
    2. Which term converts a body-frame derivative to an inertial derivative?
    3. Do Euler-angle derivatives generally equal angular-velocity components?
    4. What must be supplied before marker coordinates determine a joint moment?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Checkpoint answers': mo.md(r"""
    1. Rotation along a principal axis, or an isotropic inertia tensor.
    2. $\boldsymbol\omega\times\mathbf H$.
    3. No; they must be mapped using the chosen rotation sequence.
    4. Segment geometry/inertias, a frame convention, derivative estimates and all relevant external loads.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1) Compute the derivative of the angular momentum of the foot and the leg using one of  the following data acquired during the gait of a subject: ['../data/BiomecII2018_gait_d.txt'](../data/BiomecII2018_gait_d.txt) or ['../data/BiomecII2018_gait_n.txt'](../data/BiomecII2018_gait_n.txt).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Final answers to original Problems': mo.md(r"""
    1. For each segment, the required laboratory-frame derivative is $\dot H_G=R[I_B\dot\omega_B+\omega_B\times(I_B\omega_B)]$. The marker files alone do not determine a unique numerical answer: specify segment mass/inertia, anatomical-frame construction, sampling interval and the differentiation/filtering procedure. The two recordings produce different time series. The final result is a three-component torque-unit series (N m), not simply componentwise $I_B\dot\omega_B$.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Use [KineticsFundamentalConcepts](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinetics_fundamental_concepts.py) to connect this guide with the surrounding mechanics and modeling material. Return to the assumptions behind each equation when applying it to a new experiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional textbook investigations

    These original external assignments require the stated edition and its figures. They are retained as optional reading; the self-contained Problems above have answer panels.

    2) Problem 20.2.7 from Ruina and Rudra book.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Video lectures on the Internet

    [MIT OpenCourseWare](https://ocw.mit.edu/courses/2-003sc-engineering-dynamics-fall-2011/pages/free-body-diagrams-and-fictitious-forces/) — Free-body diagrams: practice selecting a body and frame before writing force and moment equations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Beer, F P; Johnston, E R; Cornwell, P. J.(2010) Vector Mechanics for Enginners: Dynamics.

    - Kane T, Levinson D (1985) [Dynamics: Theory and Applications](https://ecommons.cornell.edu/handle/1813/638). McGraw-Hill, Inc

    - Hibbeler, R. C. (2005) Engineering Mechanics: Dynamics.

    - Taylor, J, R (2005) Classical Mechanics

    - Winter D. A., (2009) Biomechanics and motor control of human movement. John Wiley and Sons.

    - Santos DA, Fukuchi CA, Fukuchi RK, Duarte M. (2017) A data set with kinematic and ground reaction forces of human balance. PeerJ Preprints.

    - Ruina A., Rudra P. (2019) [Introduction to Statics and Dynamics](https://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.
    """)
    return


if __name__ == "__main__":
    app.run()
