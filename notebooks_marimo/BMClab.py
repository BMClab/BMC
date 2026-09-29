import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Laboratório de Biomecânica e Controle Motor
    **[BMClab](https://bmclab.pesquisa.ufabc.edu.br/)@[UFABC](https://www.ufabc.edu.br/): Why, How, What For?**

    <br>
    <div class='center-align'><figure><img src="https://bmclab.pesquisa.ufabc.edu.br//wp-content/uploads/2016/05/cropped-BMClab0.png" alt="BMClab image header"/></figure></div>
    """)
    return


@app.cell
def _():
    from datetime import datetime
    print(datetime.now().strftime("%I:%M %p, %A, %B %d, %Y"))
    print('Marcos Duarte, https://bmclab.pesquisa.ufabc.edu.br/')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stuff in this talk

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

    180-m$^2$laboratory organized in spaces for:
    - Data collection with motion capture system, force plates, etc.
    - Data analysis with several computers
    - Subject preparation and evaluation
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

    The [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) (will) offer services for:

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
    **[Access our GitHub repository here](https://github.com/demotu/BMC).**
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
    ### Literate computing with [Jupyter Notebook](https://jupyter.org/)

    > The Jupyter Notebook is a web application that allows you to create and share documents that contain live code, equations, visualizations and explanatory text. [[jupyter.org](https://jupyter.org/)]

    See examples in [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questions?

    > [https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > E-mail: bmc.ufabc@gmail.com<br>
    > Tel.: +55 11 2320-6435<br>
    > [Location Map](https://www.google.com.br/maps/place/Federal+University+of+ABC+-+UFABC/@-23.6803572,-46.5647898,14z/data=!4m2!3m1!1s0x0:0xf1a53d9732f7a8c6)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## About these slides

    **This document (the webpage version or the slides version) is a *notebook* written using the [Jupyter Notebook](https://jupyter.org/).**

    > The Jupyter Notebook is a web application that allows you to create and share documents that contain live code, equations, visualizations and explanatory text. [[jupyter.org](https://jupyter.org/)]
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
