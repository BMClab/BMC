# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spatial and temporal characteristics of a movement pattern

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Two people cover ten metres in the same time. Have they walked in the same way? We will turn that question into measurements of steps, strides, support phases and dimensionless speed. Start with a stopwatch and a sketch of successive foot contacts; you do not need calculus.

    Try each Exercise before opening its worked solution. The checkpoint separates definitions you can measure from interpretations you cannot infer from a stopwatch. The original Problems remain practical investigations: their answer panel gives acceptable outcomes and measurement limits, not invented observations.

    **Opening challenge.** One walker takes 16 steps in 8 s over 10 m; another takes 20. Predict which has the greater cadence, step length and speed.

    ## Python setup

    This guide needs only Marimo for reading and expandable answers. Calculations can be done with a calculator; the definitions apply equally to a spreadsheet or a Python analysis.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The measurement of spatial and temporal characteristics of a movement pattern is an important resource for the characterization of an observed movement. Such variables are typically the first description performed in gait analysis (Whittle, 2007) and also used in the study of other movements and biological systems. The determination of such variables is also an excellent and valuable example of the application of relatively simple concepts of kinematics.

    Gait is the pattern of movement with the limbs by animals during terrestrial locomotion and for humans, gait consists of walking or running. Gait is typically a repetitive task where the overall pattern of movement repeats after a certain period or cycle. In the context of human gait, the movement of interest, walking or running, can be defined by steps or strides performed with the limbs. A step is the movement of one foot in front of the other and a stride is two consecutive steps with alternation of the limbs, as illustrated next.<br>
    <br>
    <div class='center-align'><figure aria-describedby="figure-caption-1"><img src="https://github.com/BMClab/BMC/blob/master/images/gaitstepstride.png?raw=1"alt="Gait step and stride"/></figure></div>

    <span id="figure-caption-1" class="figure-caption" style="display:block;text-align:center;font-style:italic">Step and stride in human gait.</span>
    <br />
    <div class='center-align'><figure aria-describedby="figure-caption-2"><img src="https://github.com/BMClab/BMC/blob/master/images/gaitcycle.png?raw=1" width=720 alt="Gait cycle"/></figure></div>

    <span id="figure-caption-2" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. The gait cycle of walking and its subphases ([http://www.gla.ac.uk/t4/~fbls/files/fab/](http://www.gla.ac.uk/t4/~fbls/files/fab/)). HS: heel strike. TO: toe off.</span>
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** Use the opening challenge to compute each walker’s speed, cadence and mean step length.

    /// details | Worked solution — Exercise 1

    Both speeds are $10/8=1.25$ m/s. The first cadence is $16/8\times60=120$ steps/min and mean step length is $10/16=0.625$ m. The second values are 150 steps/min and 0.5 m. Equal speed does not imply equal gait.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Common measurements of spatial and temporal characteristics

    The most commonly investigated spatial and temporal characteristics in the context of human gait (walking or running) analysis are:

    - Step length: distance in the direction of progression between two consecutive similar events with the different limbs. For instance, the distance between two heel strikes, with the right and left limbs.<br>
    - Stride length: distance in the direction of progression between two consecutive similar events with the same limb. For instance, the distance between two heel strikes with the right limb.<br>
    - Step duration: time duration of the step.<br>
    - Stride duration: time duration of the stride.<br>
    - Cadence: number of steps per unit of time.<br>
    - Velocity: traveled distance divided by the spent time.<br>
    - Stance duration: time duration which one limb is in contact with the ground.
    - Swing duration: time duration which one limb is not in contact with the ground.<br>
    - Single support duration: time duration which only one limb is in contact with the ground.<br>
    - Double support duration: time duration which the two limbs are in contact with the ground.<br>
    - Base of support width: distance in the frontal plane between the two feet when they were in contact with the ground (in different instants of time).

    Some of these variables can be normalized by a parameter to take into account individual characteristics or to simply make them dimensionless, for instance:

    - Dimensionless stride length: stride length divided by lower limb length.
    - Dimensionless speed: $v/\sqrt{gL}$. In the convention commonly used for locomotion, the Froude number is its square, $Fr=v^2/(gL)$; some sources use the unsquared definition, so always state the formula. Here $g$ is gravitational acceleration and $L$ is lower-limb length. See [dynamic similarity during human running](https://www.sciencedirect.com/science/article/pii/S0021929008005848).
    - Dimensionless stride frequency: stride frequency multiplied by $\sqrt{L/g}$.
    - Duty factor: period of contact of one limb with the ground divided by the stride period.
    - Stance, swing, single support and double support durations can be expressed as a fraction of the stride duration and multiplied by 100 for percentage values.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** A stride lasts 1.2 s, with 0.72 s of stance. Find swing duration, duty factor and cadence for alternating steps.

    /// details | Worked solution — Exercise 2

    Swing lasts $1.2-0.72=0.48$ s, duty factor is $0.72/1.2=0.60$, and cadence is $2(60)/1.2=100$ steps/min. The stance percentage is 60%; a single limb’s stance duration is not the total double-support duration.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Examples of use of spatial and temporal characteristics

    - The article [Bipedal animals, and their differences from humans](http://onlinelibrary.wiley.com/doi/10.1111/j.0021-8782.2004.00289.x/abstract) by Alexander describes an interesting comparison of the spatial and temporal characteristics, as well as other biomechanical and physiological variables, of gaits by humans and other animals. Alexander found a lot of similarities across animals, particularly concerning the spatial and temporal characteristics, but at the same time he concludes that no animal walks or runs as we do. See for example the article [Kinematics of bipedal locomotion while carrying a load in the arms in bearded capuchin monkeys (Sapajus libidinosus)](https://www.sciencedirect.com/science/article/abs/pii/S0047248412001807).

    - With aging, typically it's observed a decrease in gait speed, an increase in double stance time, and an increase in step width, among other changes. See the website [Gait Disorders in the Elderly](http://www.merckmanuals.com/professional/geriatrics/gait_disorders_in_the_elderly/gait_disorders_in_the_elderly.html) for more details.

    - A study involving 26,802 individuals 60 years and older from 17 different countries found that the simple measurement of gait speed, combined with reported cognitive complaints, was associated with increased future dementia risk at the population level; it does not establish which individual will develop dementia ([Verghese et al., 2014](https://www.neurology.org/doi/10.1212/WNL.0000000000000717)).

    - See the article [Clinical Assessment of Spatiotemporal Gait Parameters in Patients and Older Adults](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4353430/) reporting the use of a photoelectric system to measure spatial and temporal characteristics.

    - There is a classic article published in 1976 showing that the walking speed of pedestrians is positively correlated with the size of the city! (see [The Pace of Life](https://www.nature.com/articles/259557a0)). The authors interpreted the higher walking speed of people in larger cities as a psychological response to stimulatory overload. However, a later article offer a trivial explanation for this finding (see [The Pace of Life - Reanalysed: Why Does Walking Speed of Pedestrians Correlate with City Size?](https://www.jstor.org/stable/4535062)).
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** At 1.5 m/s with a 0.9 m leg, calculate dimensionless speed and the squared Froude convention.

    /// details | Worked solution — Exercise 3

    $v/\sqrt{gL}=1.5/\sqrt{9.81(0.9)}=0.505$ and $Fr=v^2/(gL)=0.255$. The two numbers describe the same normalization but must not be interchanged without stating their definitions.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example of a clinical gait analysis

    <img src="https://github.com/BMClab/BMC/blob/master/images/walking_animation.gif?raw=1" alt="walking" width="480" height="480" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### [Sample gait analysis report](https://bmclab.pesquisa.ufabc.edu.br/wp-content/uploads/2016/08/SampleReportWalking.pdf)
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Can mean speed distinguish short rapid steps from long slow steps?
    2. Which event pair defines a stride?
    3. Which support variables require observing both feet?

    /// details | Checkpoint answers

    1. No; speed combines step length and step frequency. 2. Successive equivalent contacts of the same foot. 3. Single and double support require the contact state of both feet.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Propose different instruments to measure spatial and temporal characteristics of the human gait.
    2. Design and perform a simple experiment to measure as many spatial and temporal characteristics as possible during walking at different speeds (slow, normal, and fast) using only a chronometer and a known distance of about 10 m. Compare your results with the data from [Spatial and Temporal Descriptors](https://clinicalgate.com/kinesiology-of-walking/#s0020).
    3. Use a video camera (or propose and use another method) to measure the characteristics you couldn't measure in the previous item.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Final answers to the problems

    1. Stopwatch plus measured walkway: mean speed, step count and cadence. Calibrated video: step/stride lengths and event timing. Pressure walkway or instrumented insoles: contact timing and placement. A force plate: contact forces and timing within its footprint.
    2. Measured outcomes vary. Report trial distance, elapsed time, number of steps, speed, cadence and mean step length for each speed, with start/stop rules and repeated-trial variability. A stopwatch alone does not identify individual stance, swing or double-support phases.
    3. Video of both feet with known frame rate gives heel-contact and toe-off times; a calibrated floor view also gives step lengths and width. Timing resolution is limited by the frame interval and event visibility. Report the observed values and uncertainty, not assumed normative percentages.

    ///
    """)
    return









@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Read [KinematicsParticle](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_particle.py) for kinematic definitions and sampling.

    ### Video lectures on the Internet

    [MIT Classical Mechanics](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/) has short lessons on kinematics, circular motion and Newton’s laws. Choose the lesson that matches the measurement or balance equation used here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Alexander RM (2004) [Bipedal animals, and their differences from humans](http://onlinelibrary.wiley.com/doi/10.1111/j.0021-8782.2004.00289.x/abstract). Journal of Anatomy, 204, 5, 321-330.

    - [Gait Disorders in the Elderly](http://www.merckmanuals.com/professional/geriatrics/gait_disorders_in_the_elderly/gait_disorders_in_the_elderly.html).

    - Verghese J et al. (2014) [Motoric cognitive risk syndrome](https://www.neurology.org/doi/10.1212/WNL.0000000000000717). Neurology, doi: 10.1212/WNL.0000000000000717.

    - Whittle M (2007) [Gait Analysis: An Introduction](https://books.google.com.br/books?id=HtNqAAAAMAAJ). Butterworth-Heinemann.
    """)
    return


if __name__ == "__main__":
    app.run()
