import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Projectile motion

    > Marcos Duarte,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    This notebook takes what is probably the simplest interesting problem in mechanics — a body moving through the air with nothing but gravity acting on it — and pushes it until it says something honest about human jumping. Along the way it predicts an optimum take-off angle of $45^o$, and then confronts that prediction with the fact that no long jumper on Earth takes off anywhere near $45^o$.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a number *before* the code prints it; the prediction is the point, and being wrong is the most useful thing that can happen to you here.

    The mathematics is deliberately plain: it is the constant-acceleration case from the notebook [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb), done twice — once for the horizontal direction, where the acceleration is zero, and once for the vertical, where it is $-g$. Everything else in this notebook follows from those two lines.

    **Challenge 0.** Before you begin, think of one jump or throw you care about: a jump shot, a shot put, a dog catching a frisbee, your own vertical jump. Write down how high you think it goes and how long you think it stays in the air. Keep it nearby; the last sections ask you to come back to it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A jump worth understanding

    <div style="background-color:#F1F1F1;border:1px solid black;padding:10px;">
        <figure><img src="https://bmclab.pesquisa.ufabc.edu.br/x/salto2/jadel.gif" alt="triple jump"/><br><figcaption><center><i>Figure. Animations of a triple jump performed by athlete Jadel Gregório during training. The jump distance was 16.20 meters. The two animations are synchronized and represent the same athlete during the same jump, with the one on the left representing the athlete's movement subtracting the horizontal displacement of his center of gravity. From the website <a href="https://bmclab.pesquisa.ufabc.edu.br/analise-biomecanica-do-salto-em-distancia/">Análise do salto em distância</a>.</i></center></figcaption></figure>
    </div>

    Watch the animation on the right for a while before reading on. The athlete is doing a great deal: swinging his arms, cycling his legs, rotating his trunk. And yet, between each take-off and each landing, there is one thing about him that is entirely out of his control.

    That thing is the path of his center of gravity. Once his foot leaves the ground, no arm swing and no leg cycle can change where that point goes; they can only change how the body is arranged *around* it. The animation on the left makes the point in another way: with the horizontal travel of the center of gravity subtracted, what is left is a man running on the spot.

    This notebook is about that uncontrollable path.

    **Guiding questions 0.**

    1. If the athlete cannot change his center of gravity's trajectory in the air, what exactly is all that arm and leg movement for?
    2. He covered 16.20 m in three contacts. Guess the flight time of one of them. You will be able to check your guess in a few sections.
    3. A jump of 16.20 m from a body whose legs are barely a metre long: where does the distance come from?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    Two libraries carry this notebook: NumPy for the numbers and Matplotlib for the plots.
    """)
    return


@app.cell
def _():
    import numpy as np

    import matplotlib

    matplotlib.rc("axes", labelsize=13, titlesize=14)
    matplotlib.rc("xtick", labelsize=11)
    matplotlib.rc("ytick", labelsize=11)
    matplotlib.rc("legend", fontsize=11)
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model, and what it throws away

    **Projectile** or **ballistic motion** is the movement of a body in the air near Earth's surface where the main force acting on the body is the gravitational force. If we neglect the air resistance and approximate the gravity force (which acts only in the vertical direction) on the body as constant, the body will have constant velocity in the horizontal direction and constant acceleration (the gravitational acceleration) in the vertical direction.

    That sentence hides three decisions, and it is worth naming them:

    1. **The body is a particle.** Its size, shape and orientation are ignored; only one point moves. For a body made of many parts, that point is the *center of mass*, and we return to this below.
    2. **There is no air resistance.** The only force is gravity. This is the assumption that fails first, and how badly it fails depends entirely on what you threw.
    3. **Gravity is constant**, $g = 9.8\;\mathrm{m/s^2}$, pointing down, everywhere along the path.

    None of the three is true. All three are close enough for a human jump, which is why the model earns its place in biomechanics.

    **Challenge 1.** Rank these by how badly assumption 2 breaks: a shot put, a javelin, a badminton shuttlecock, a long jumper, a table tennis ball with topspin. For the two worst offenders, say *why* — is it the speed, the mass, the surface area, or something the object is doing on purpose?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Equations of motion

    Let's deduce the equations for a projectile motion and investigate their characteristics. For simplicity, let's consider for now a planar movement, neglect the air resistance, and take the gravitational acceleration as $g=9.8\;\mathrm{m/s^2}$. Consider a body (which we will treat as a particle) launched into the air with initial angle $\theta$ with respect to the horizontal (the $\mathbf{x}$ direction) and initial velocity $\mathbf{v_0}$, a vector quantity with components $v_{0x}=v_0\cos(\theta)$ and $v_{0y}=v_0\sin(\theta)$.

    The equations of motion for position ($x, y$), velocity ($v_x, v_y$), and acceleration ($a_x, a_y$) are:

    $$
    \begin{align}
    & x(t) = x_0 + v_0\cos(\theta)\:t \\
    & y(t) = y_0 + v_0\sin(\theta)\:t - \frac{g\:t^2}{2} \\
    & v_x(t) = v_0\cos(\theta) \\
    & v_y(t) = v_0\sin(\theta) - g\:t \\
    & a_x(t) = 0 \\
    & a_y(t) = -g
    \end{align}
    $$

    Read the two columns of that block separately and the whole thing becomes much smaller than it looks. The horizontal column is *a particle at constant speed*. The vertical column is *a particle at constant acceleration*. They are the second and third of the three standard cases derived in [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb), and they do not talk to each other: gravity never appears in $x(t)$, and the launch angle never appears in $a_y(t)$.

    That independence is the single most useful fact about projectile motion. A bullet fired horizontally and a bullet dropped from the same height hit the ground at the same instant.

    Everything in the next four sections is obtained by asking the same two questions of these equations: *when is the vertical velocity zero* (the top of the flight), and *when is the vertical position back to where it started* (the landing).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Time of flight

    The time of flight can be calculated from the equation for the vertical velocity, using the properties that at the maximum height the vertical velocity is zero and that the time of rising is equal to the time of falling:

    $$
    t_{flight} = \frac{2v_0\sin(\theta)}{g}
    $$

    Note what is *not* in that expression: the horizontal velocity. How long a projectile stays in the air is decided entirely at take-off by its vertical velocity, and running faster does not buy you one millisecond more.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Maximum height

    Using the time of flight we just calculated, and once again the fact that at the maximum height the body has zero vertical velocity, the maximum height $h$ can be obtained from the equation for the vertical position:

    $$
    h = \frac{v_0^2\sin^2(\theta)}{2g}
    $$

    The square on $v_0$ is worth a moment. Ten percent more vertical velocity at take-off buys about twenty-one percent more height, which is why vertical-jump training is so obsessive about the instant of take-off and so indifferent to everything that happens afterwards.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Range, the maximum horizontal distance

    The range, $R$, is the maximum horizontal distance reached with respect to the point of release. The horizontal motion has constant velocity, so it is simply the horizontal velocity multiplied by the time available:

    $$
    R = v_0\cos(\theta)\:t_{flight}
    $$

    Substituting the expression for $t_{flight}$:

    $$
    R = \frac{v_0^2\sin(2\theta)}{g}
    $$

    This compact form carries the whole tension of a jump. The angle appears once, inside $\sin(2\theta)$, which is largest at $\theta = 45^o$. The speed appears squared, so it always helps. Distance is a negotiation between going fast and going up, and the two are in conflict because they share the same take-off velocity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parabolic trajectory

    The equation for the horizontal position versus time is the equation of a straight line, and for the vertical position it is the equation of a parabola. We can prove that the spatial trajectory (the vertical position versus the horizontal position) is also a parabola if we use the two equations for position and eliminate the variable time:

    $$
    y = y_0 + \tan(\theta)(x-x_0) - \frac{g}{2v_0^2\cos^2(\theta)}(x-x_0)^2
    $$

    Since $y_0,\: x_0,\: \theta,\: v_0,\: g$ are constants, this is the equation of a parabola.

    **Guiding questions 1.**

    1. Two projectiles are launched with the same speed, one at $30^o$ and the other at $60^o$. Which lands farther? Which stays in the air longer? Which is higher at the top?
    2. The range equation says $R$ grows with $v_0^2$ but the time of flight grows only with $v_0$. Explain the difference in one sentence.
    3. On the Moon, $g$ is about six times smaller. By what factor does the range of the same jump change? And the maximum height?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Seeing the trajectories

    Let's compute instead of only deriving. The function below returns the positions of a projectile released with a given speed and angle, from a given height, until it comes back down to zero height. It is nothing more than the two position equations above, plus the landing time obtained by solving the vertical equation for $y = 0$.

    **Before you run the next cells**, predict two things and write them down:

    1. Among the release angles $15^o$, $30^o$, $45^o$, $60^o$ and $75^o$, which goes farthest, and by how much does it beat the second?
    2. Two of those five angles land at almost exactly the same distance. Which two?
    """)
    return


@app.function
def trajectory(v0, theta, h0=0.0, g=9.8, n=200):
    """Trajectory of a projectile released with speed `v0` [m/s] at `theta` [deg].

    The body is released from the height `h0` [m] and the trajectory ends when
    it returns to the height zero. Returns the instants [s] and the horizontal
    and vertical positions [m].
    """
    import numpy as np

    vx = v0 * np.cos(np.deg2rad(theta))
    vy = v0 * np.sin(np.deg2rad(theta))
    t_flight = (vy + np.sqrt(vy**2 + 2 * g * h0)) / g
    t = np.linspace(0, t_flight, n)

    return t, vx * t, h0 + vy * t - g * t**2 / 2


@app.function
def flight(v0, theta, h0=0.0, g=9.8):
    """Time of flight [s], horizontal distance [m] and peak height [m].

    For a projectile released with speed `v0` [m/s] at the angle `theta` [deg]
    from the height `h0` [m], landing at the height zero.
    """
    import numpy as np

    vx = v0 * np.cos(np.deg2rad(theta))
    vy = v0 * np.sin(np.deg2rad(theta))
    t_flight = (vy + np.sqrt(vy**2 + 2 * g * h0)) / g

    return t_flight, vx * t_flight, h0 + vy**2 / (2 * g)


@app.function
def plot_trajectories(v0, angles, h0=0.0, g=9.8):
    """Plot the trajectories of a projectile released at different angles."""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(9, 4.5))
    for _theta in angles:
        _, _x, _y = trajectory(v0, _theta, h0=h0, g=g)
        ax.plot(_x, _y, linewidth=2, label=f"{_theta:.0f}$^o$")
        ax.plot(_x[-1], _y[-1], "o", color=ax.lines[-1].get_color())

    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel("Horizontal position [m]")
    ax.set_ylabel("Vertical position [m]")
    ax.set_title(f"Projectile motion, $v_0$ = {v0} m/s, $h_0$ = {h0} m")
    ax.legend(title="Release angle", loc="upper right")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()


@app.cell
def _():
    plot_trajectories(v0=10, angles=[15, 30, 45, 60, 75])
    return


@app.cell
def _():
    print("Angle   Time of flight   Distance   Peak height")
    for _theta in [15, 30, 45, 60, 75, 90]:
        _t, _d, _h = flight(v0=10, theta=_theta)
        print(f"{_theta:4.0f}° {_t:12.2f} s {_d:10.2f} m {_h:11.2f} m")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Release angle for maximum horizontal distance

    From the expression for the range, we note that the maximum value is reached when $\sin(2\theta)=1$, which gives $\theta=45^o$. The figure and the table confirm it.

    They also show the symmetry that the algebra was hiding. Because $\sin(2\theta) = \sin(180^o - 2\theta)$, any two angles that add up to $90^o$ give exactly the same range: $30^o$ and $60^o$ land in the same place, and so do $15^o$ and $75^o$. They get there by very different routes — one flat and quick, one high and slow — but they get there.

    ### Release angle for maximum height

    From the expression for the maximum height, we note that the maximum value is reached when $\sin^2(\theta)=1$, which gives $\theta=90^o$. Straight up. Obvious once stated, and the last row of the table is the reminder of what it costs: a range of zero.

    **Challenge 2.** The curve of range against angle is *flat* near its maximum: going from $45^o$ to $40^o$ costs surprisingly little distance. Use the `flight` function to work out how much you lose, in percent, at $40^o$ and at $35^o$. Keep the answer in mind — the long jump section is about to make it matter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Playing with projectile motion

    Here is an interactive animation of a projectile motion where you can change the initial angle and speed of the body (or, to start, simply press the button *Fire* in the rectangular menu). Try to reproduce the symmetry you just found: fire at $30^o$, then at $60^o$, with the same speed. And then turn on air resistance and watch the symmetry break.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        """<iframe
        src="https://phet.colorado.edu/sims/html/projectile-motion/latest/projectile-motion_en.html"
        width="100%" height="500" style="border:none"></iframe>"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kinematics of the long jump

    We can study the kinematics of jumps by humans and other animals as projectile motions with the equations of motion we derived, if we consider the body as a particle. There is a particular point in the body (a virtual point) that will strictly follow those equations. This point is called the **center of mass**, or center of gravity, and represents an average position of all the masses in the body. We will study it later; for now just consider that when applying the equations for projectile motion to a body with many particles, we are in fact describing the motion of its center of mass.

    This is the formal version of what the triple-jump animation showed. The athlete's limbs are not projectiles. One imaginary point inside him is.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The different phases of the long jump

    The track and field event long jump has some particularities we must consider to analyze it as a projectile motion; the figure below shows the kinematic characteristics relevant for this analysis.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/longjump.png?raw=1" width=700 alt="longjump"/></center><figcaption><center><i>Figure. Diagram of a long jump and its kinematic characteristics. The red circle represents the center of mass of the jumper. Adapted from Linthorne (2007).</i></center></figcaption></figure>

    The distance officially considered as the jump distance, $d_{jump}$, can be described as the sum of three distances as indicated in the figure:

    $$
    d_{jump} = d_{take-off} + d_{flight} + d_{landing}
    $$

    Only the middle term is projectile motion. The other two are bookkeeping, and they are where a competition is quietly won or lost: $d_{take-off}$ is how far in front of the board the center of mass already is when the foot leaves it, and $d_{landing}$ is how far in front of the center of mass the heels are allowed to reach before the body follows.

    **Guiding questions 2.**

    1. The figure shows the center of mass *higher* at take-off than at landing. Why, given that both happen at ground level?
    2. Which of the three distances would you try to improve if you were coaching a jumper who keeps fouling at the board?
    3. Is $d_{landing}$ limited by physics or by technique?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The distance of flight

    The jump in the air is the part of the jump that we can describe as a projectile motion, and its distance, $d_{flight}$, is typically 90% of the jump distance for professional jumpers.

    Let's derive an expression for $d_{flight}$, which we called range before, as a function of the velocity and angle at take-off. Looking at the long jump diagram, we see that the jumper takes off from a higher height than when they land. The equation for the vertical displacement of the jumper's center of mass during the flight phase will be:

    $$
    h_{landing} = h_{take-off} + v_0\sin(\theta)\:t - \frac{g\:t^2_{flight}}{2}
    $$

    For simplicity let's adopt $h_0=h_{take-off}-h_{landing}$:

    $$
    0 = h_0 + v_0\sin(\theta)\:t - \frac{g\:t^2_{flight}}{2}
    $$

    Solving this second-order equation for $t_{flight}$:

    $$
    t_{flight} = \frac{v_0 \sin(\theta) \pm \sqrt{v^2_0 \sin^2(\theta) + 2 h_0 g}}{g}
    $$

    This expression is different from the one we derived before because the initial and final heights are different in the long jump. However, it reduces to the former expression when this difference is zero, $h_0=0$.

    We are only interested in the positive solution of the equation (the negative solution results in a negative time), and substituting this result in the equation for the horizontal distance we have an expression for $d_{flight}$:

    $$
    d_{flight} = \frac{v_0\cos(\theta)\left[v_0 \sin(\theta) + \sqrt{v^2_0\sin^2(\theta) + 2 h_0 g}\:\right]}{g}
    $$

    Once again, when $h_0=0$, we have the same expression we derived before. This is exactly the formula already implemented in the `flight` function above — the `h0` argument we have been leaving at its default.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The optimum angle of take-off

    We can now find the angle that maximizes the range given that the projectile is released (the body jumps) from an initial height. To find such an angle, we differentiate the previous equation with respect to the angle, set this derivative equal to zero, and solve it (read about how to find maxima points in a function [here](http://en.wikipedia.org/wiki/Maxima_and_minima)). It can be shown that the solution is (for a proof, see [here](http://math.stackexchange.com/questions/127300/maximum-range-of-a-projectile-launched-from-an-elevation) or [here](http://www.themcclungs.net/physics/download/H/2_D_Motion/Projectile%20Cliff.pdf)):

    $$
    \theta = \arctan\left(\frac{v_0}{\sqrt{v_0^2 + 2gh_0}}\right)
    $$

    For $h_0=0$ the angle is $\arctan(1)$, which is $45^o$, as obtained before. But the higher the release, the smaller the fraction, and the smaller the optimum angle. Falling further than you rose is a free extension of the flight, and it is bought more cheaply with horizontal speed than with height.

    #### Computation of the optimum angle

    Let's write a function to calculate the optimum angle given the initial height and velocity:
    """)
    return


@app.function
def optimum_angle(height, velocity, g=9.8):
    """Optimum angle [deg] for maximum range of a projectile.

    For a projectile released from `height` [m] with speed `velocity` [m/s].
    """
    import numpy as np

    if velocity == 0:
        return 0.0

    return float(
        np.rad2deg(np.arctan(velocity / np.sqrt(velocity**2 + 2 * g * height)))
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's determine the optimum angle for an initial height of 0.6 m and an initial velocity of 9 m/s — actual values for long jump athletes — and see what that angle is worth in metres:
    """)
    return


@app.cell
def _():
    jump_height = 0.6  # difference in height of the center of mass, m
    jump_velocity = 9  # take-off speed, m/s
    jump_angle = optimum_angle(jump_height, jump_velocity)

    print(
        f"The optimum angle for a jump with h = {jump_height:.1f} m and "
        f"v = {jump_velocity:.1f} m/s is {jump_angle:.1f} degrees"
    )
    for _theta in [jump_angle, 45, 30, 20]:
        _, _d, _ = flight(jump_velocity, _theta, h0=jump_height)
        print(f"  take-off at {_theta:4.1f} degrees  ->  d_flight = {_d:.2f} m")
    return (jump_height,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can even plot the optimum angle for different values of initial height and velocity (e.g., heights from 0 to 1 m and velocities from 5 to 10 m/s):
    """)
    return


@app.function
def plot_optimum_angle(heights, velocities):
    """Surface of the optimum release angle vs initial height and velocity."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    angles = np.zeros((len(heights), len(velocities)))
    for i, _h in enumerate(heights):
        for j, _v in enumerate(velocities):
            angles[i, j] = optimum_angle(height=_h, velocity=_v)

    h, v = np.meshgrid(heights, velocities, indexing="ij")

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(
        h,
        v,
        angles,
        rstride=1,
        cstride=1,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    ax.set_xlim(heights[0], heights[-1])
    ax.set_ylim(velocities[0], velocities[-1])
    ax.set_zlim(np.min(angles), 45)
    ax.set_xlabel("Initial height (m)", fontsize=12)
    ax.set_ylabel("Initial velocity (m/s)", fontsize=12)
    ax.set_zlabel("Optimum angle ($^o$)", fontsize=12)
    ax.set_title("Projectile motion: optimum angle for maximum range", fontsize=16)
    plt.show()


@app.cell
def _(np):
    plot_optimum_angle(
        heights=np.arange(0, 1.05, 0.05),
        velocities=np.arange(5, 10.25, 0.25),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The whole surface lives between about $37^o$ and $45^o$. Over the entire range of heights and speeds a human being can produce, the physics never asks for an angle far from $45^o$.

    Hold on to that, because it is about to be contradicted by every long jumper who has ever competed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Actual take-off angles from human jumps are different from the prediction

    The typical angles of take-off among athletes vary between $15^o$ and $27^o$ (Linthorne, 2007) — much lower than we predicted. Neither the physics nor the simplifications we adopted are wrong.

    What happens is that humans are unable to run horizontally at 10 m/s and suddenly change their direction to $45^o$ without losing a large amount of that speed. The contact time of the foot with the ground just before jumping, for a long jump athlete, is about 10 ms. Our muscles do not have the capacity to act fast enough and generate the necessary amount of force in such a short time.

    Look carefully at what this changes in the problem. Our derivation assumed $v_0$ and $\theta$ were *independent*: you pick an angle, and the speed is whatever it is. For a real jumper they are not independent at all. Every degree of take-off angle is paid for in take-off speed, and the optimum we computed is the optimum of a choice nobody gets to make.

    **Before you run the next cells**, predict: if each extra degree of take-off angle costs the athlete a tenth of a metre per second of take-off speed, roughly where does the best angle end up?
    """)
    return


@app.function
def plot_takeoff_tradeoff(v_max, h0, costs, theta_max=60):
    """Flight distance vs take-off angle when speed is traded for angle.

    The take-off speed is modelled as ``v0 = v_max - cost*theta``, a crude
    stand-in for the fact that a jumper cannot redirect the run-up upwards for
    free. `costs` are the speed penalties [m/s per degree]; `cost = 0` is the
    independent case assumed in the derivations above. Prints and returns the
    best angle and distance for each cost.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    theta = np.linspace(0, theta_max, 601)
    best = []

    _, ax = plt.subplots(figsize=(9, 4.5))
    for cost in costs:
        v0 = np.clip(v_max - cost * theta, 0, None)
        d = np.array([flight(_v, _th, h0=h0)[1] for _v, _th in zip(v0, theta)])
        i = int(d.argmax())
        best.append((cost, theta[i], d[i]))
        ax.plot(theta, d, linewidth=2, label=f"{cost:.2f} m/s per degree")
        ax.plot(theta[i], d[i], "o", color=ax.lines[-1].get_color())

    ax.axvspan(15, 27, color="0.85", zorder=0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.text(21, ax.get_ylim()[1] * 0.04, "observed", ha="center", fontsize=11)
    ax.set_xlabel("Take-off angle [$^o$]")
    ax.set_ylabel("Flight distance [m]")
    ax.set_title(f"Cost of the angle, $v_{{max}}$ = {v_max} m/s, $h_0$ = {h0} m")
    ax.legend(title="Speed penalty", loc="upper right")
    plt.tight_layout()
    plt.show()

    print("Speed penalty      Best angle   Take-off speed   Flight distance")
    for cost, theta_best, d_best in best:
        print(
            f"{cost:5.2f} m/s per degree {theta_best:9.1f}° "
            f"{v_max - cost * theta_best:13.2f} m/s {d_best:12.2f} m"
        )

    return best


@app.cell
def _(jump_height):
    _ = plot_takeoff_tradeoff(v_max=10.0, h0=jump_height, costs=[0.0, 0.05, 0.10, 0.15])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With no penalty the optimum sits at the $43^o$ we derived. A penalty of a tenth of a metre per second per degree — a small number, and not an implausible one — drags the optimum straight into the band that athletes actually use, and the curve there is so flat that the exact angle hardly matters.

    Be clear about what that model is and is not. The linear penalty is a caricature, invented here to make one point visible; the real relationship between take-off speed and take-off angle has been measured, and it is the subject of Linthorne (2007). But the shape of the conclusion survives the caricature: **once the take-off speed depends on the take-off angle, the optimum angle collapses**, and it collapses to roughly where human beings jump.

    So don't conclude that the physics of projectile motion is useless for understanding human jumps — quite the contrary. It is possible to model the mechanical and physiological properties of the human body and simulate a long jump, using these same equations of motion, and reproduce most of the characteristics of a real jump; see, for example, Alexander (1990) and Seyfarth et al. (2000). We can even use this kind of modeling to ask what would happen if an athlete changed some part of their technique.

    See experimental values for the kinematic properties of long jumps measured during competitions on the websites [Research Projects](http://www.iaaf.org/development/research) (from the International Association of Athletics Federations) and [Análise Biomecânica do Salto em Distância](http://demotu.org/x/salto/) (in Portuguese).

    **Challenge 3.** Find, by trial and error with the function above, the speed penalty that puts the optimum at exactly $20^o$. Then answer the harder question: how would you *measure* that penalty on a real athlete, rather than assume it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## On the calculation of the angle from kinematic data

    Everything above assumed the take-off angle was given. In an experiment it has to be measured, and measuring it is less innocent than it looks.

    The angle of the projectile with the horizontal (in the context of a human jump, the angle of the trajectory of the center of gravity with the horizontal in the sagittal plane) at a given instant is, by definition, the arc whose tangent is the ratio of the vertical ($y$) and horizontal ($x$) displacements:

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/segment.png?raw=1" width=250 alt="segment"/></center><figcaption><center><i>Figure. A segment in a plane and its coordinates.</i></center></figcaption></figure>

    Given the coordinates at two instants:

    $$
    \theta = \arctan\left(\frac{y_2-y_1}{x_2-x_1}\right)
    $$

    A simple way to determine the angle with experimental data, rather than calculating the finite differences of position in the two directions, is to calculate the angle from the ratio of the vertical and horizontal velocities, $v(t)$, since:

    $$
    \theta(t) = \arctan\left(\frac{v_y(t)}{v_x(t)}\right) =
    \arctan\left(\dfrac{\dfrac{y_{t}-y_{t-1}}{\Delta t}}{\dfrac{x_{t}-x_{t-1}}{\Delta t}}\right) =
    \arctan\left(\frac{y_{t}-y_{t-1}}{x_{t}-x_{t-1}}\right)
    $$

    The $\Delta t$ cancels, which is convenient and slightly misleading: it suggests the sampling rate does not matter. It does. The ratio of finite differences is the *average* direction over the interval, and the direction is changing throughout it — that is what gravity does. So the angle you recover is always the angle of a chord, never of the tangent at take-off.

    **Before you run the next cell**, predict: a jumper really takes off at $20^o$; you film them at 30 Hz and compute the angle from the first two frames. How far off will you be — a hundredth of a degree, a tenth, or a whole degree?
    """)
    return


@app.function
def angle_from_samples(v0, theta, rates, h0=0.0, g=9.8):
    """Take-off angle [deg] recovered by finite differences at several rates.

    Samples the true trajectory of a projectile released at `theta` [deg] with
    speed `v0` [m/s] at each sampling rate in `rates` [Hz], then estimates the
    angle from the first two samples, as one would from filmed data. Returns a
    list of (rate, estimated angle, error).
    """
    import numpy as np

    out = []
    for rate in rates:
        dt = 1 / rate
        t = np.array([0, dt])
        x = v0 * np.cos(np.deg2rad(theta)) * t
        y = h0 + v0 * np.sin(np.deg2rad(theta)) * t - g * t**2 / 2
        estimated = float(np.rad2deg(np.arctan2(y[1] - y[0], x[1] - x[0])))
        out.append((rate, estimated, estimated - theta))

    return out


@app.cell
def _():
    print("True take-off angle: 20.0°\n")
    print("Sampling rate   Estimated angle   Error")
    for _rate, _est, _err in angle_from_samples(
        v0=9, theta=20, rates=[30, 60, 120, 240, 1000]
    ):
        print(f"{_rate:8.0f} Hz {_est:15.2f}° {_err:8.2f}°")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The error is always negative — the estimate is always too shallow — because the projectile is losing vertical velocity during the interval and gaining none horizontally. It is a *bias*, not noise: averaging more trials will not remove it, and it shrinks only as the sampling interval shrinks.

    At 30 Hz, a standard video camera, you underestimate a $20^o$ take-off by about a degree. Whether that matters depends on what you do with the number. If you feed it into the range equation to predict the jump, it is negligible. If you are comparing two athletes whose take-off angles differ by $2^o$, you have just spent half of your effect on the frame rate.

    This is the same lesson that the 100 m analysis in [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb) arrives at from the other direction: **a finite difference cannot resolve anything faster than its own interval**, and every derivative you take from measured positions is a compromise between that blur and the noise you amplify by shortening the interval. See the notebooks on [data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb) and [residual analysis](https://github.com/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb).

    **Challenge 4.** Modify `angle_from_samples` to estimate the angle from a *centered* difference — the samples just before and just after take-off — instead of a forward one. Does the bias get better, worse, or disappear? Can you say why before you run it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. A jumper wants to stay in the air longer. What exactly must change at take-off, and what will it cost them in distance?
    2. Why is the time of flight independent of the horizontal velocity, while the range is not?
    3. We predicted an optimum take-off angle near $45^o$ and observed $15^o$–$27^o$. Was the prediction wrong, was the measurement wrong, or was the question wrong?
    4. Two athletes take off with the same speed, one at $22^o$ and the other at $25^o$. Without computing, is the difference in their flight distance likely to be larger or smaller than the difference in their take-off angles suggests?
    5. You have video of a jump at 30 Hz and you need the take-off velocity vector. Name two sources of error in the result and say which one you could reduce by buying a better camera.
    6. The center of mass follows a parabola in flight. A gymnast performing a somersault clearly does not. Is the model broken?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. An athlete during a long jump had at the moment of takeoff a velocity of (8.10, 3.58, 0) [in m/s], where $\mathbf{x, y, z}$ are the horizontal (anterior-posterior), vertical, and medio-lateral directions, respectively.<br>
       a. Calculate the magnitude of the velocity at the takeoff.<br>
       b. Calculate the angle of the jump at the takeoff.<br>
       c. Calculate the jump distance.<br>
       d. The actual jump distance was 6.75 m. Comment about the difference between this value and the distance found in (c).

    2. A person throws a ball upward (vertically) into the air with an initial speed of 10 m/s and initial height of release of 2 m. Determine the following quantities for the ball:<br>
       a. Maximum height.<br>
       b. Total time in the air.<br>
       c. Velocity just before impact with the ground.<br>
       d. Plots for position, velocity, and acceleration.

    3. An athlete of diving jumps from a 10-m height platform and reaches the water at a horizontal distance of 5 m after 2.5 s. Calculate the following quantities for the diver's center of mass:<br>
       a. Initial speed and angle at the takeoff.<br>
       b. Maximum height reached.

    4. A ball is thrown with a speed $v_0=25$ m/s and angle $\theta=\tan^{-1}(4/3)$ and strikes the inclined surface shown in the figure below. Determine the position $s$ at which the ball will strike the inclined surface. Solution: $s = 40.04$ m.

        <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/ex1_12_rade.png?raw=1" width=400 alt="inclined surface"/></center></figure>

    5. Investigate and propose at least three different methods to measure these kinematic quantities during training and sporting events. Discuss the limitations and advantages of each method considering their application in different sports.

    6. The website [Physics Classroom](https://www.physicsclassroom.com/curriculum/vectors/Projectile-Motion) has a link to a pdf document with 26 problems about projectile motion and links to the lessons about this topic. Download the pdf and solve the problems.

    7. Return to the jump or throw you chose in Challenge 0. Estimate its release speed and angle from your own guesses about its height and range, using the equations in this notebook. Then say which of the three modelling assumptions in *The model, and what it throws away* is the one you would have to abandon first to do better.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## More examples

    - [Horizontally Launched Projectiles — Problem-Solving](https://www.physicsclassroom.com/class/vectors/Lesson-2/Horizontally-Launched-Projectiles-Problem-Solving)
    - [Non-Horizontally Launched Projectiles — Problem-Solving](https://www.physicsclassroom.com/class/vectors/Lesson-2/Non-Horizontally-Launched-Projectiles-Problem-Solv)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - Read the text: Linthorne NP (2007) <a href="http://www.brunel.ac.uk/~spstnpl/Publications/Ch24LongJump(Linthorne).pdf">Biomechanics of the long jump</a>. In Routledge Handbook of Biomechanics and Human Movement Science, Y. Hong and R. Bartlett (Editors), Routledge, London. pp. 340–353. It measures the speed–angle trade-off we only caricatured here.
    - [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb) — where the constant-acceleration equations used here come from.
    - [Scalar and vector](https://github.com/BMClab/BMC/blob/master/notebooks/ScalarVector.ipynb) — how to handle the velocity vector at take-off in Python.
    - [Center of mass and moment of inertia](https://github.com/BMClab/BMC/blob/master/notebooks/CenterOfMassAndMomentOfInertia.ipynb) — the point that actually follows the parabola.
    - [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb) and [residual analysis](https://github.com/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb) — on getting velocities out of measured positions.

    ### Video lectures on the Internet

    - Khan Academy: [Two-dimensional motion](https://www.khanacademy.org/science/ap-physics-1/ap-two-dimensional-motion)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Alexander RM (1990) [Optimum take-off techniques for high and long jumps](http://rstb.royalsocietypublishing.org/cgi/pmidlookup?view=long&pmid=1976267). Philosophical Transactions of the Royal Society of London. Series B, Biological Sciences, 329(1252), 3-10.
    - [Análise Biomecânica do Salto em Distância](http://demotu.org/x/salto/)
    - Linthorne NP (2007) <a href="http://www.brunel.ac.uk/~spstnpl/Publications/Ch24LongJump(Linthorne).pdf">Biomechanics of the long jump</a>. In Routledge Handbook of Biomechanics and Human Movement Science, Y. Hong and R. Bartlett (Editors), Routledge, London. pp. 340–353.
    - [Research Projects](http://www.iaaf.org/development/research) from the International Association of Athletics Federations.
    - Seyfarth A, Blickhan R, Van Leeuwen JL (2000) [Optimum take-off techniques and muscle design for long jump](http://jeb.biologists.org/cgi/pmidlookup?view=long&pmid=10648215). Journal of Experimental Biology, 203(Pt 4), 741-50.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
