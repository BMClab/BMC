# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "ipython"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Biomechanics and Motor Control Laboratory

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil

    **BMClab @ UFABC: Why, How, What For?**

    ![BMClab](https://bmclab.pesquisa.ufabc.edu.br/wp-content/uploads/2016/05/cropped-BMClab0.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A laboratory is more than a list of instruments. Its research question determines which measurements matter and how other people can check the result. This guide preserves the original BMClab presentation, including its Portuguese title and English discussion, and uses it to plan a small movement study. Equipment, funding, and service lists are historical descriptions from that presentation; consult the lab for current availability.

    Read the examples in order, predict each result, and then run the cell. Try the distributed Exercises before opening their worked solutions. Finish with the checkpoint questions and independent Problems; their panels contain final answers only.

    **Challenge 0.** You want to study standing balance. Which would you record first: body motion, ground forces, or muscle activity? State what each could tell you and what it could not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    This overview needs Marimo and IPython for the original video embeds. Images and videos require Internet access. No laboratory data are downloaded and no equipment is controlled.
    """)
    return


@app.cell
def _():
    from datetime import datetime
    print(datetime.now().strftime("%I:%M %p, %A, %B %d, %Y"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The questions behind the laboratory

    - [BMClab](http://demotu.org): why and how
    - [BMClab](http://demotu.org) infrastructure, current and future activities
    - Open data science and education
    - Literate programming & Literate computing
    - Advocacy for Python (the programming language)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) website

    [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why the [BMClab](https://bmclab.pesquisa.ufabc.edu.br/)

    **[Biomedical engineering](https://en.wikipedia.org/wiki/Biomedical_engineering)**: the application of engineering principles and design concepts to medicine and biology for healthcare *and well-being* purposes.

    **[Neuroscience of human movement](https://en.wikipedia.org/wiki/Neuroscience)**: the scientific study of the nervous system bases of controlling movement.

    **[Biomechanics](https://en.wikipedia.org/wiki/Biomechanics) and [Motor Control](https://en.wikipedia.org/wiki/Motor_control)**: the study of the structure and function of biological systems using the knowledge and methods of the Mechanics and the study of how the biological systems control their movements.

    **[BMClab](http://demotu.org)**: In a broad sense, we are interested in knowing how living beings control and execute their movements. We also work to improve the quality of life in society by offering evaluation services in our laboratory and in the dissemination of scientific knowledge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) lines of research

    - Postural control in humans
    - Clinical gait analysis
    - Biomechanics of long distance running
    - Modeling and simulation of the neuromusculoskeletal system
    - ...
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** A project records center-of-pressure motion while a person stands quietly. Is that automatically a direct measurement of neural control?

    /// details | Solution to Exercise 1
    No. It is a mechanical output of the person–ground interaction. Neural control is one contributor, together with body mechanics and the task. A control hypothesis needs a model and suitable comparisons, not just a measured trajectory.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) financial support

    The [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) was made possible by the financial support from UFABC and from Brazilian research agencies, nominally:
    - Project "Controle do equilíbrio e movimento em adultos jovens e idosos sedentários e corredores" (FAPESP).
    - Project "Postura e envelhecimento: criação de base de dados pública de sinais de oscilação e simulação computacional de mecanismos de controle" (FAPESP).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) financial support (II)

    The [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) was made possible by the financial support from UFABC and from Brazilian research agencies, nominally:
    - Project "Estudo do equilíbrio de pessoas com deficiências e idosos: uma base de dados aberta" (CNPq).
    - Project "Desenvolvimento de simulador de cadeira de rodas e de serviço de avaliação do movimento e postura para deficientes físicos com próteses/órteses e usuários de cadeira de rodas" (MCTI-SECIS/CNPq).
    - Project "Análise de atletas corredores: estudo multicêntrico para compreensão do movimento com implicação para prevenção de lesão e melhora do rendimento" (ME/CNPq).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) infrastructure

    180-m$^2$ laboratory organized in spaces for:<br>
    - Data collection with motion capture system, force plates, etc.<br>
    - Data analysis with several computers<br>
    - Subject preparation and evaluation<br>
    - Machine and electronics assembly
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div class='center-align'><figure><img src="https://bmclab.pesquisa.ufabc.edu.br/wp-content/uploads/2016/01/BMClab4.png" alt="BMClab lab"/></figure></div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) equipment

    - 12-camera Motion capture system
    - Six-component instrumented dual-belt treadmill
    - Six-component force plates
    - Pressure distribution transducers
    - Tri-axial accelerometers
    - Wheelchair six-component transducers
    - Six-component torque transducer
    - 10-channel wireless electromyography system
    - ...
    """)
    return


@app.cell
def _():
    from IPython.display import YouTubeVideo
    YouTubeVideo('1ZwYlaqvCSw', width=800, height=480, rel=0)
    return (YouTubeVideo,)


@app.cell
def _(YouTubeVideo):
    YouTubeVideo('tp_rP9C0ysY', width=800, height=480, rel=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) services

    The original presentation proposed the following services; this historical list is not a current booking catalogue:

    - Clinical gait analysis
    - Running biomechanics assessment
    - Wheelchair propulsion assessment
    - General motion capture
    - ...
    """)
    return


@app.cell
def _(YouTubeVideo):
    YouTubeVideo('5ZKMVWkOyZA', width=800, height=480, rel=0)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** Choose two instruments from the equipment list for a gait study and explain why neither alone identifies every muscle force.

    /// details | Solution to Exercise 2
    Motion capture estimates segment kinematics and a force plate records external contact forces. Together with inertial assumptions they support inverse dynamics, but a net joint moment can be produced by many muscle-force combinations. Additional modeling or measurements are needed.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Open data science

    > Open science data is a type of open data focused on publishing observations and results of scientific activities available for anyone to analyze and reuse. [[wikipedia.org](https://en.wikipedia.org/wiki/Open_science_data)]

    **The [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) is committed to open science data.**<br>
    **[Access our data here](https://bmclab.pesquisa.ufabc.edu.br/datasets/).**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Open education

    > Open education is a collective term to describe institutional practices and programmatic initiatives that broaden access to the learning and training traditionally offered through formal education systems. [[wikipedia.org](https://en.wikipedia.org/wiki/Open_education)]

    **The [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) is committed to open education.**<br>
    **[Access our GitHub repository here](https://github.com/BMClab/BMC).**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Literate programming and literate computing

    > Literate programming: Instead of imagining that our main task is to instruct a computer what to do, let us concentrate rather on explaining to human beings what we want a computer to do. [[Donald Knuth (1984)](http://www.literateprogramming.com/knuthweb.pdf)]


    > Literate computing: A literate computing environment is one that allows users not only to execute commands **interactively** but also to store in a literate document format the results of these commands along with figures and free-form text that can include formatted mathematical expressions. [[Millman KJ and Perez F (2014)](https://osf.io/h9gsd/?action=download&version=1)]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Literate computing with [marimo](https://marimo.io/)

    A marimo notebook combines executable Python, equations, visualizations, and explanatory text. Its reactive execution keeps dependent calculations synchronized when an input changes.

    Explore the guides in the [BMC collection](https://github.com/BMClab/BMC#readme).
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** What should accompany a public data file so another student can reproduce a figure?

    /// details | Solution to Exercise 3
    Supply variable definitions, units, sampling information, processing steps, relevant consent/access conditions, a data version, and executable analysis with its package environment. A downloadable array without context is not yet a reproducible analysis.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questions?

    > [BMClab website](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > E-mail: bmc.ufabc@gmail.com<br>
    > Tel.: +55 11 2320-6435<br>
    > [Location Map](https://www.google.com.br/maps/place/Federal+University+of+ABC+-+UFABC/@-23.6803572,-46.5647898,14z/data=!4m2!3m1!1s0x0:0xf1a53d9732f7a8c6)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## About these slides

    **The original presentation was written in Jupyter. This edition is a reactive Marimo notebook; both formats combine explanation, executable calculations, and media.**

    Open this edition locally in marimo or use the cloud links in the repository's README.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. What connects biomechanics and motor control?
    2. Does open data mean unlabeled data?
    3. What distinguishes a notebook from a static screenshot?

    /// details | Checkpoint answers
    1. Mechanical movement and the biological processes regulating it.
    2. No; reusable data require context and appropriate access conditions.
    3. A notebook exposes an executable calculation and its explanation.
    ///

    ## Problems

    1. Propose one measurable question about standing balance.
    2. Name two metadata items needed for a force-platform record.
    3. Does the historical equipment list establish that an instrument is available today?

    /// details | Final answers
    1. Example: does center-of-pressure variability change between eyes-open and eyes-closed standing?
    2. Sampling frequency and force units; coordinate convention is another valid answer.
    3. No.
    ///

    ## Go deeper

    [Biomechanics](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/biomechanics.py) — choose a mechanical question. [BMClab datasets](https://bmclab.pesquisa.ufabc.edu.br/datasets/) — consult the dataset description and availability before planning an analysis.

    ### Video lectures on the Internet

    [MIT Linear Algebra lectures](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/video_galleries/video-lectures/) — vectors and orthogonality give the mathematical language for describing measured motion.

    ## References

    - Original BMClab presentation and [laboratory website](https://bmclab.pesquisa.ufabc.edu.br/). Knuth (1984), [Literate Programming](http://www.literateprogramming.com/knuthweb.pdf). Millman and Pérez (2014), original literate-computing reference retained above.
    """)
    return


if __name__ == "__main__":
    app.run()
