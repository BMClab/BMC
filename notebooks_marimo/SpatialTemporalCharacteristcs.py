import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Spatial and temporal characteristics of a movement pattern

        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Common-measurements-of-spatial-and-temporal-characteristics" data-toc-modified-id="Common-measurements-of-spatial-and-temporal-characteristics-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Common measurements of spatial and temporal characteristics</a></span></li><li><span><a href="#Examples-of-use-of-spatial-and-temporal-characteristics" data-toc-modified-id="Examples-of-use-of-spatial-and-temporal-characteristics-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Examples of use of spatial and temporal characteristics</a></span><ul class="toc-item"><li><span><a href="#Example-of-a-clinical-gait-analysis" data-toc-modified-id="Example-of-a-clinical-gait-analysis-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Example of a clinical gait analysis</a></span><ul class="toc-item"><li><span><a href="#Sample-gait-analysis-report" data-toc-modified-id="Sample-gait-analysis-report-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span><a href="https://bmclab.pesquisa.ufabc.edu.br/wp-content/uploads/2016/08/SampleReportWalking.pdf" rel="nofollow" target="_blank">Sample gait analysis report</a></a></span></li></ul></li></ul></li><li><span><a href="#Problems" data-toc-modified-id="Problems-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The measurement of spatial and temporal characteristics of a movement pattern is an important resource for the characterization of an observed movement. Such variables are typically the first description performed in gait analysis (Whittle, 2007) and also used in the study of other movements and biological systems. The determination of such variables is also an excellent and valuable example of the application of relatively simple concepts of kinematics.

        Gait is the pattern of movement with the limbs by animals during terrestrial locomotion and for humans, gait consists of walking or running. Gait is typically a repetitive task where the overall pattern of movement repeats after a certain period or cycle. In the context of human gait, the movement of interest, walking or running, can be defined by steps or strides performed with the limbs. A step is the movement of one foot in front of the other and a stride is two consecutive steps with alternation of the limbs, as illustrated next.  
        <br>
        <div class='center-align'><figure><img src="./../images/gaitstepstride.png"alt="Gait step and stride"/><figcaption><center><i>Step and stride in human gait.</i></center></figcaption></figure></div> 
        <br />
        <div class='center-align'><figure><img src="./../images/gaitcycle.png" width=720 alt="Gait cycle"/><figcaption><center><i>Figure. The gait cycle of walking and its subphases ([http://www.gla.ac.uk/t4/~fbls/files/fab/](http://www.gla.ac.uk/t4/~fbls/files/fab/)). HS: heel strike. TO: toe off.</i></center></figcaption></figure></div> 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Common measurements of spatial and temporal characteristics

        The most commonly investigated spatial and temporal characteristics in the context of human gait (walking or running) analysis are:

        - Step length: distance in the direction of progression between two consecutive similar events with the different limbs. For instance, the distance between two heel strikes, with the right and left limbs.   
        - Stride length: distance in the direction of progression between two consecutive similar events with the same limb. For instance, the distance between two heel strikes with the right limb.   
        - Step duration: time duration of the step.   
        - Stride duration: time duration of the stride.     
        - Cadence: number of steps per unit of time.   
        - Velocity: traveled distance divided by the spent time.    
        - Stance duration: time duration which one limb is in contact with the ground.
        - Swing duration: time duration which one limb is not in contact with the ground.   
        - Single support duration: time duration which only one limb is in contact with the ground.  
        - Double support duration: time duration which the two limbs are in contact with the ground.    
        - Base of support width: distance in the frontal plane between the two feet when they were in contact with the ground (in different instants of time).

        Some of these variables can be normalized by a parameter to take into account individual characteristics or to simply make them dimensionless, for instance:   

        - Dimensionless stride length: stride length divided by lower limb length.
        - Dimensionless speed: the [Froude number](http://en.wikipedia.org/wiki/Froude_number),$v/\sqrt{gL}$, where g is the gravitational acceleration and L is the lower limb length.
        - Dimensionless stride frequency: stride frequency multiplied by$\sqrt{L/g}$.
        - Duty factor: period of contact of one limb with the ground divided by the stride period.
        - Stance, swing, single support and double support durations can be expressed as a fraction of the stride duration and multiplied by 100 for percentage values.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Examples of use of spatial and temporal characteristics

        - The article [Bipedal animals, and their differences from humans](http://onlinelibrary.wiley.com/doi/10.1111/j.0021-8782.2004.00289.x/abstract) by Alexander describes an interesting comparison of the spatial and temporal characteristics, as well as other biomechanical and physiological variables, of gaits by humans and other animals. Alexander found a lot of similarities across animals, particularly concerning the spatial and temporal characteristics, but at the same time he concludes that no animal walks or runs as we do. See for example the article [Kinematics of bipedal locomotion while carrying a load in the arms in bearded capuchin monkeys (Sapajus libidinosus)](https://www.sciencedirect.com/science/article/abs/pii/S0047248412001807).

        - With aging, typically it's observed a decrease in gait speed, an increase in double stance time, and an increase in step width, among other changes. See the website [Gait Disorders in the Elderly](http://www.merckmanuals.com/professional/geriatrics/gait_disorders_in_the_elderly/gait_disorders_in_the_elderly.html) for more details.

        - A study involving 26,802 individuals 60 years and older from 17 different countries found that the simple measurement of gait speed, combined with a kind of memory test, can be used to identify high-risk seniors that will develop dementia ([Verghese et al., 2014](http://www.neurology.org/content/83/8/718)).  

        - See the article [Clinical Assessment of Spatiotemporal Gait Parameters in Patients and Older Adults](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4353430/) reporting the use of a photoelectric system to measure spatial and temporal characteristics.  

        - There is a classic article published in 1976 showing that the walking speed of pedestrians is positively correlated with the size of the city! (see [The Pace of Life](https://www.nature.com/articles/259557a0)). The authors interpreted the higher walking speed of people in larger cities as a psychological response to stimulatory overload. However, a later article offer a trivial explanation for this finding (see [The Pace of Life - Reanalysed: Why Does Walking Speed of Pedestrians Correlate with City Size?](https://www.jstor.org/stable/4535062)).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example of a clinical gait analysis

        <img src="https://bmclab.pesquisa.ufabc.edu.br/wp-content/uploads/2016/08/walkAnimation.gif" alt="walking" width="480" height="480" />
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### [Sample gait analysis report](https://bmclab.pesquisa.ufabc.edu.br/wp-content/uploads/2016/08/SampleReportWalking.pdf)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. Propose different instruments to measure spatial and temporal characteristics of the human gait.  
        2. Design and perform a simple experiment to measure as many spatial and temporal characteristics as possible during walking at different speeds (slow, normal, and fast) using only a chronometer and a known distance of about 10 m. Compare your results with the data from [Spatial and Temporal Descriptors](https://clinicalgate.com/kinesiology-of-walking/#s0020). 
        3. Use a video camera (or propose and use another method) to measure the characteristics you couldn't measure in the previous item.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Alexander RM (2004) [Bipedal animals, and their differences from humans](http://onlinelibrary.wiley.com/doi/10.1111/j.0021-8782.2004.00289.x/abstract). Journal of Anatomy, 204, 5, 321-330.  
        - [Gait Disorders in the Elderly](http://www.merckmanuals.com/professional/geriatrics/gait_disorders_in_the_elderly/gait_disorders_in_the_elderly.html).  
        - Verghese J et al. (2014) [Motoric cognitive risk syndrome](http://www.neurology.org/content/83/8/718). Neurology, doi: 10.1212/WNL.0000000000000717.
        - Whittle M (2007) [Gait Analysis: An Introduction](http://books.google.com.br/books?id=HtNqAAAAMAAJ). Butterworth-Heinemann.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
