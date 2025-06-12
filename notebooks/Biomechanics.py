import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to Biomechanics

        > Marcos Duarte, Renato Naville Watanabe  
        > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br)  
        > Federal University of ABC, Brazil  

        <p style="text-align: right;">A <a href="https://jupyter.org/">Jupyter Notebook</a></p>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Biomechanics-@-UFABC" data-toc-modified-id="Biomechanics-@-UFABC-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Biomechanics @ UFABC</a></span></li><li><span><a href="#Biomechanics" data-toc-modified-id="Biomechanics-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Biomechanics</a></span><ul class="toc-item"><li><span><a href="#Biomechanics-&amp;-Mechanics---Hatze" data-toc-modified-id="Biomechanics-&amp;-Mechanics---Hatze-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Biomechanics &amp; Mechanics - Hatze</a></span></li><li><span><a href="#Biomechanics-vs.-Mechanics---Hatze" data-toc-modified-id="Biomechanics-vs.-Mechanics---Hatze-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Biomechanics vs. Mechanics - Hatze</a></span></li><li><span><a href="#Biomechanics-&amp;-Mechanics---Fung" data-toc-modified-id="Biomechanics-&amp;-Mechanics---Fung-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Biomechanics &amp; Mechanics - Fung</a></span></li><li><span><a href="#Branches-of-Mechanics" data-toc-modified-id="Branches-of-Mechanics-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Branches of Mechanics</a></span></li><li><span><a href="#Biomechanics-&amp;-other-Sciences-I" data-toc-modified-id="Biomechanics-&amp;-other-Sciences-I-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Biomechanics &amp; other Sciences I</a></span></li><li><span><a href="#Biomechanics-&amp;-Engineering" data-toc-modified-id="Biomechanics-&amp;-Engineering-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Biomechanics &amp; Engineering</a></span></li><li><span><a href="#Applications-of-Biomechanics" data-toc-modified-id="Applications-of-Biomechanics-2.7"><span class="toc-item-num">2.7&nbsp;&nbsp;</span>Applications of Biomechanics</a></span></li><li><span><a href="#On-the-branches-of-Mechanics-I" data-toc-modified-id="On-the-branches-of-Mechanics-I-2.8"><span class="toc-item-num">2.8&nbsp;&nbsp;</span>On the branches of Mechanics I</a></span></li><li><span><a href="#On-the-branches-of-Mechanics-and-Biomechanics" data-toc-modified-id="On-the-branches-of-Mechanics-and-Biomechanics-2.9"><span class="toc-item-num">2.9&nbsp;&nbsp;</span>On the branches of Mechanics and Biomechanics</a></span></li><li><span><a href="#The-future-of-Biomechanics" data-toc-modified-id="The-future-of-Biomechanics-2.10"><span class="toc-item-num">2.10&nbsp;&nbsp;</span>The future of Biomechanics</a></span></li><li><span><a href="#Biomechanics--and-the-Biomedical-Engineering-at-UFABC-I" data-toc-modified-id="Biomechanics--and-the-Biomedical-Engineering-at-UFABC-I-2.11"><span class="toc-item-num">2.11&nbsp;&nbsp;</span>Biomechanics  and the Biomedical Engineering at UFABC I</a></span></li><li><span><a href="#Biomechanics--and-the-Biomedical-Engineering-at-UFABC-II" data-toc-modified-id="Biomechanics--and-the-Biomedical-Engineering-at-UFABC-II-2.12"><span class="toc-item-num">2.12&nbsp;&nbsp;</span>Biomechanics  and the Biomedical Engineering at UFABC II</a></span></li><li><span><a href="#More-on-Biomechanics" data-toc-modified-id="More-on-Biomechanics-2.13"><span class="toc-item-num">2.13&nbsp;&nbsp;</span>More on Biomechanics</a></span></li></ul></li><li><span><a href="#History-of-Biomechanics" data-toc-modified-id="History-of-Biomechanics-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>History of Biomechanics</a></span><ul class="toc-item"><li><span><a href="#Aristotle-(384-322-BC)" data-toc-modified-id="Aristotle-(384-322-BC)-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Aristotle (384-322 BC)</a></span><ul class="toc-item"><li><span><a href="#Aristotle-&amp;-the-Scientific-Revolution-I" data-toc-modified-id="Aristotle-&amp;-the-Scientific-Revolution-I-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Aristotle &amp; the Scientific Revolution I</a></span></li><li><span><a href="#Aristotle-&amp;-the-Scientific-Revolution-II" data-toc-modified-id="Aristotle-&amp;-the-Scientific-Revolution-II-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Aristotle &amp; the Scientific Revolution II</a></span></li></ul></li><li><span><a href="#Leonardo-da-Vinci-(1452-1519)" data-toc-modified-id="Leonardo-da-Vinci-(1452-1519)-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Leonardo da Vinci (1452-1519)</a></span></li><li><span><a href="#Giovanni-Alfonso-Borelli-(1608-1679)" data-toc-modified-id="Giovanni-Alfonso-Borelli-(1608-1679)-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Giovanni Alfonso Borelli (1608-1679)</a></span></li><li><span><a href="#Eadweard-James-Muybridge-(1830–1904)-and-Étienne-Jules-Marey-(1830-1904)" data-toc-modified-id="Eadweard-James-Muybridge-(1830–1904)-and-Étienne-Jules-Marey-(1830-1904)-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Eadweard James Muybridge (1830–1904) and Étienne-Jules Marey (1830-1904)</a></span></li><li><span><a href="#More-on-the-history-of-Biomechanics" data-toc-modified-id="More-on-the-history-of-Biomechanics-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>More on the history of Biomechanics</a></span></li></ul></li><li><span><a href="#The-International-Society-of-Biomechanics" data-toc-modified-id="The-International-Society-of-Biomechanics-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>The International Society of Biomechanics</a></span></li><li><span><a href="#Biomechanics-by-(BMClab)-examples" data-toc-modified-id="Biomechanics-by-(BMClab)-examples-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Biomechanics by (BMClab) examples</a></span><ul class="toc-item"><li><span><a href="#Examples-of-Biomechanics-Classes-around-the-World" data-toc-modified-id="Examples-of-Biomechanics-Classes-around-the-World-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Examples of Biomechanics Classes around the World</a></span></li></ul></li><li><span><a href="#Biomechanics-classes-@-UFABC" data-toc-modified-id="Biomechanics-classes-@-UFABC-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Biomechanics classes @ UFABC</a></span></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-Internet" data-toc-modified-id="Video-lectures-on-the-Internet-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Video lectures on the Internet</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Biomechanics @ UFABC

        [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##  Biomechanics

        The etymology of the word *Biomechanics* is evident:

        <span class="notranslate">$\text{Biomechanics} := \text{bios} \, (\text{life}) + \text{mechanics}$</span>  

        Professor Herbert Hatze, on a letter to the editors of the Journal of Biomechanics in 1974, proposed a (very good) definition for *the science called Biomechanics*:

        >"*Biomechanics is the study of the structure and function of biological systems by means of the methods of mechanics.*"   
        >  
        > Hatze (1974) [The meaning of the term biomechanics](https://github.com/demotu/BMC/blob/master/courses/refs/HatzeJB74biomechanics.pdf).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Biomechanics & Mechanics - Hatze

        And Hatze, advocating for *Biomechanics to be a science of its own*, argues that Biomechanics **is not** simply Mechanics of (applied to) living systems:

        > "*It would not be correct to state that 'Biomechanics is the study of the mechanical aspects of the structure and function of biological systems' because biological systems do not have mechanical aspects. They only have biomechanical aspects (otherwise mechanics, as it exists, would be sufficient to describe all phenomena which we now call biomechanical features of biological systems).*" Hatze (1974)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Biomechanics vs. Mechanics - Hatze

        To support this argument, Hatze illustrates the difference between Biomechanics and the application of Mechanics, with an example of a javelin throw: studying the mechanics aspects of the javelin flight trajectory (use existing knowledge about aerodynamics and ballistics) vs. studying the biomechanical aspects of the phase before the javelin leaves the thrower’s hand (there are no established mechanical models for this system).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Biomechanics & Mechanics - Fung

        Professor Yuan-Cheng Fung, another great biomechanicist, is (much) less enthusiastic than Hatze about Biomechanics to be a science of its own, according to Fung:
        > "*Biomechanics is mechanics applied to biology*" Fung (1993).

        But this definition does not mean Fung demotes the importance of Biomechanics, a few pages later he stated:  
        > "*Biomechanics has participated in virtually every modern advance of medical science and technology*" Fung (1993).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Branches of Mechanics

        **A good knowledge of Mechanics is a necessary condition, but not sufficient!, to have a good knowledge of Biomechanics**.  

        In fact, only a subset of Mechanics matters to Biomechanics, the Classical Mechanics subset, the domain of mechanics for bodies with moderate speeds$(\ll 3.10^8 m/s!)$and not very small$(\gg 3.10^{-9} m!)$as shown in the following diagram:  

        <figure><center><img src="http://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Physicsdomains.svg/500px-Physicsdomains.svg.png" width=500 alt="Domains of mechanics"/></center><figcaption><center><i>Figure. Domains of mechanics (image from <a href="http://en.wikipedia.org/wiki/Classical_mechanics">http://en.wikipedia.org/wiki/Classical_mechanics</a>).</i></center></figcaption></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Biomechanics & other Sciences I

        One last point about the excellent letter from Hatze, already in 1974 he points for the following problem:

        > "*The use of the term biomechanics imposes rather severe restrictions on its meaning because of the established definition of the term, mechanics. This is unfortunate,  since the synonym Biomechanics, as it is being understood by the majority of biomechanists today, has a much wider meaning.*" Hatze (1974)

        Although the term Biomechanics may sound new to you, it's not rare that people think the use of methods outside the realm of Mechanics as Biomechanics.  
        For instance, electromyography and thermography are two methods that although may be useful in Biomechanics, particularly the former, they clearly don't have any relation with Mechanics; Electromagnetism and Thermodynamics are other [branches of Physics](https://en.wikipedia.org/wiki/Branches_of_physics), although there is considerable overlapping between Mechanics and Thermodynamics.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Biomechanics & Engineering

        Even seeing Biomechanics as a field of Science, as argued by Hatze, it's also possible to refer to Engineering Biomechanics considering that  
        *Engineering is the application of scientific and mathematical principles to practical ends* [[The Free Dictionary](http://www.thefreedictionary.com/engineering)] and particularly that  
        *Engineering Mechanics is the application of Mechanics to solve problems involving common engineering elements* [[Wikibooks]](https://en.wikibooks.org/wiki/Engineering_Mechanics), and, last but not least, that  
        *Biomedical engineering is the application of engineering principles and design concepts to medicine and biology for healthcare purposes* [[Wikipedia](https://en.wikipedia.org/wiki/Biomedical_engineering)].
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Applications of Biomechanics

        Biomechanics matters to fields of science and technology related to biology and health and it's also relevant for the development of synthetic systems inspired on biological systems, as in robotics. To illustrate the variety of applications of Biomechanics, this is the current list of topics covered in the Journal of Biomechanics:
        """
    )
    return


@app.cell
def _():
    from IPython.display import IFrame
    IFrame('https://www.sciencedirect.com/journal/journal-of-biomechanics/about/aims-and-scope', width='100%', height=500)
    return (IFrame,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### On the branches of Mechanics I

        Mechanics is a branch of the physical sciences that is concerned with the state of rest or motion of bodies that are subjected to the action of forces. In general, this subject can be subdivided into three branches: rigid-body mechanics, deformable-body mechanics, and fluid mechanics (Hibbeler, 2012; Ruina and Rudra, 2019).

        (Classical) Mechanics is typically partitioned in Statics and Dynamics (Hibbeler, 2012; Ruina and Rudra, 2019).  
        In turn, Dynamics is divided in **Kinematics** and **Kinetics**.  
        This classification is clear; dynamics is the study of the motions of bodies and Statics is the study of forces in the absence of changes in motion. Kinematics is the study of motion without considering its possible causes (forces) and Kinetics is the study of the possible causes of motion.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <figure><center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Mechanics_Overview_Table.jpg/640px-Mechanics_Overview_Table.jpg" width=800 alt="Branches of mechanics"/></center><figcaption><center><i>Figure. Branches of mechanics (image from <a href="https://en.wikibooks.org/wiki/Engineering_Statics/Introduction">https://en.wikibooks.org/wiki/Engineering_Statics/Introduction</a>).</i></center></figcaption></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### On the branches of Mechanics and Biomechanics

        Nevertheless, it's common in Biomechanics to adopt a slightly different classification: to partition it between Kinematics and Kinetics, and then Kinetics into Statics and Dynamics (David Winter, Nigg & Herzog, and Vladimir Zatsiorsky, among others, use this classification in their books). The rationale is that we first separate the study of motion considering or not its causes (forces). The partition of (Bio)Mechanics in this way is useful because is simpler to study and describe (measure) the kinematics of human motion and then go to the more complicated issue of understanding (measuring) the forces related to the human motion.

        Anyway, these different classifications reveal a certain contradiction between Mechanics (particularly from an engineering point of view) and Biomechanics; some scholars will say that this taxonomy in Biomechanics is simply wrong and it should be corrected to align with the Mechanics. Be aware.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The future of Biomechanics

        (Human) Movement Science combines many disciplines of science (such as, Physiology, Biomechanics, and Psychology) for the study of human movement. Professor Benno Nigg claims that with the growing concern for the well-being of humankind, Movement Science will have an important role:
        > Movement science will be one of the most important and most recognized science fields in the twenty-first century... The future discipline of movement science has a unique opportunity to become an important contributor to the well-being of mankind.   
        Nigg BM (1993) [Sport  science  in  the twenty-first  century](http://www.ncbi.nlm.nih.gov/pubmed/8230394). Journal of Sports Sciences, 77, 343-347.

        And so Biomechanics will also become an important contributor to the well-being of humankind.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Biomechanics  and the Biomedical Engineering at UFABC I

        At the university level, the study of Mechanics is typically done in the disciplines Statics and Dynamics (rigid-body mechanics), Strength of Materials (deformable-body mechanics), and Mechanics of Fluids (fluid mechanics).  
        Consequently, the study on Biomechanics must also cover these topics for a greater understanding of the structure and function of biological systems.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Biomechanics  and the Biomedical Engineering at UFABC II

        The Biomedical Engineering degree at UFABC covers these topics for the study of biological systems in different courses: Ciência dos Materiais Biocompatíveis, Modelagem e Simulação de Sistemas Biomédicos, Métodos de Elementos Finitos aplicados a Sistemas Biomédicos, Mecânica dos Fluidos, Caracterização de Biomateriais, Sistemas Biológicos, and last but not least, Biomecânica I, Biomecânica II & Modelagem e simulação do movimento humano (Biomecãnica III).  

        How much of biological systems is in fact studied in these disciplines varies a lot. Anyway, none of these courses cover the study of human motion with implications to health, rehabilitation, and sports, except the last course. This is the reason why the courses Biomecânica I & II focus on the analysis of the human movement.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### More on Biomechanics

        The Wikipedia page on biomechanics is a good place to read more about Biomechanics:
        """
    )
    return


@app.cell
def _(IFrame):
    IFrame('https://en.m.wikipedia.org/wiki/Biomechanics', width='100%', height=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## History of Biomechanics

        Biomechanics progressed basically with the advancements in Mechanics and with the invention of instrumentations for measuring mechanical quantities and computing.  

        The development of Biomechanics was only possible because people became more interested in the understanding of the structure and function of biological systems and to apply these concepts to the progress of the humankind.  

        Thus, it is natural to associate the great minds of Mechanics, Galileo, Newton, Euler, Lagrange, among others, as contributors to the advancement of Biomechanics.

        Let's look briefly at some direct contributors to Biomechanics.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Aristotle (384-322 BC)  
        Aristotle was the first to have written about the movement of animals in his works *On the Motion of Animals (De Motu Animalium)* and *On the Gait of Animals (De Incessu Animalium)* [[Works by Aristotle]](http://classics.mit.edu/Browse/index-Aristotle.html).

        Aristotle clearly already knew what we nowadays refer as Newton's third law of motion:  
        "*For as the pusher pushes so is the pushed pushed, and with equal force.*" [Part 3, [On the Motion of Animals](http://classics.mit.edu/Aristotle/motion_animals.html)]
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Aristotle & the Scientific Revolution I

        Although Aristotle's contributions were invaluable to humankind, to make his discoveries he doesn't seem to have employed anything similar to what we today refer as [scientific method](https://en.wikipedia.org/wiki/Scientific_method) (the systematic observation, measurement, and experiment, and the formulation, testing, and modification of hypotheses).

        Most of the Physics of Aristotle was ambiguous or incorrect; for example, for him there was no motion without a force. He even deduced that speed was proportional to force and inversely proportional to resistance [[Book VII, Physics](http://classics.mit.edu/Aristotle/physics.7.vii.html)]. Perhaps Aristotle was too influenced by the observation of motion of a body under the action of a friction force, where this notion is not at all unreasonable.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Aristotle & the Scientific Revolution II

        If Aristotle performed any observation/experiment at all in his works, he probably was not good on that as, ironically, evinced in this part of his writing:  
        > "Males have more teeth than females in the case of men, sheep, goats, and swine; in the case of other animals observations have not yet been made". Aristotle [The History of Animals](http://classics.mit.edu/Aristotle/history_anim.html).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Leonardo da Vinci (1452-1519)

        <div><figure><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Da_Vinci_Vitruve_Luc_Viatour.jpg/353px-Da_Vinci_Vitruve_Luc_Viatour.jpg' width="240" alt="Vitruvian Man" style="float:right;margin: 0 0 0 20px;"/></figure></div>

        Contributions of Leonardo to Biomechanics:  
         - Studies on the proportions of humans and animals  
         - Anatomy studies of the human body, especially the foot  
         - Studies on the mechanical function of muscles  

        <br><br>
        Figure. *"Le proporzioni del corpo umano secondo Vitruvio", also known as the [Vitruvian Man](https://en.wikipedia.org/wiki/Vitruvian_Man), drawing by [Leonardo da Vinci](https://en.wikipedia.org/wiki/Leonardo_da_Vinci) circa 1490 based on the work of [Marcus Vitruvius Pollio](https://en.wikipedia.org/wiki/Vitruvius) (1st century BC), depicting a man in supposedly ideal human proportions (image from [Wikipedia](https://en.wikipedia.org/wiki/Vitruvian_Man)).*  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Giovanni Alfonso Borelli (1608-1679)

        <div><figure><img src='https://upload.wikimedia.org/wikipedia/commons/d/d5/Giovanni_Borelli_-_lim_joints_%28De_Motu_Animalium%29.jpg' width="240" alt="Borelli" style="float:right;margin: 0 0 0 20px;"/></figure></div>

         - [The father of biomechanics](https://en.wikipedia.org/wiki/Giovanni_Alfonso_Borelli); the first to use modern scientific method into 'Biomechanics' in his book [De Motu Animalium](http://www.e-rara.ch/doi/10.3931/e-rara-28707).  
         - Proposed that the levers of the musculoskeletal system magnify motion rather than force.  
         - Calculated the forces required for equilibrium in various joints of the human body  before Newton published the laws of motion.  
        <br><br>
        Figure. *Excerpt from the book De Motu Animalium*.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Eadweard James Muybridge (1830–1904) and Étienne-Jules Marey (1830-1904)

         - [Eadweard Muybridge](https://en.wikipedia.org/wiki/Eadweard_Muybridge): development of photography for movement analysis.  
         - [Étienne-Jules Marey](https://en.wikipedia.org/wiki/%C3%89tienne-Jules_Marey): development of first devices for measuring foot–ground contact forces.  

        <center>
        <div><figure><img src='https://upload.wikimedia.org/wikipedia/commons/d/d2/The_Horse_in_Motion_high_res.jpg' width="400" alt="Borelli" style="float:left;"/></figure></div>
        <div><figure><img src='https://upload.wikimedia.org/wikipedia/commons/2/2d/Animal_mechanism-_a_treatise_on_terrestrial_and_a%C3%ABrial_locomotion_%281874%29_%2818198040265%29.jpg' width="400" alt="Borelli" style="float:right;"/></figure></div>
        </center>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### More on the history of Biomechanics

        See:  
        - [A Genealogy of Biomechanics ](https://courses.washington.edu/bioen520/notes/History_of_Biomechanics_(Martin_1999).pdf)    
        - [History of Biomechanics and Kinesiology](https://biomechanics.vtheatre.net/doc/history.html)  
        - Chapter 1 of Nigg and Herzog (2006) [Biomechanics of the Musculo-skeletal System](https://books.google.com.br/books?id=hOIeAQAAIAAJ&dq=editions:ISBN0470017678)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The International Society of Biomechanics

        The biomechanics community has an official scientific society, the [International Society of Biomechanics](http://isbweb.org/), with a journal, the [Journal of Biomechanics](http://www.jbiomech.com), and an e-mail list, the [Biomch-L](http://biomch-l.isbweb.org):
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Biomechanics by (BMClab) examples

        Biomechanics has been applied to many different problems; let's see a few examples of applications of Biomechanics:  
        - Clinical gait analysis: [https://bmclab.pesquisa.ufabc.edu.br/servicos/cga/](https://bmclab.pesquisa.ufabc.edu.br/servicos/cga/)  
        - Biomechanics of sports: [https://bmclab.pesquisa.ufabc.edu.br/biomechanics-of-the-bicycle-kick/](https://bmclab.pesquisa.ufabc.edu.br/biomechanics-of-the-bicycle-kick/)  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Examples of Biomechanics Classes around the World
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - [Biomechanical Engineering Courses](https://me.stanford.edu/groups/biomechanical-engineering-program/biomechanical-engineering-courses)  
        - [BME 366-0-01: Biomechanics of Movement](https://www.mccormick.northwestern.edu/biomedical/academics/courses/descriptions/366.html)  
        - [Biomechanics at MIT](https://ocw.mit.edu/search/?q=biomechanics)  
        - [Biomechanics curriculum](https://www.me.washington.edu/students/grad/curriculum/biomechanics)  
        - [Online Biomechanics Courses](https://edutestlabs.com/online-biomechanics-courses/)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Biomechanics classes @ UFABC

         - **[Biomecânica I](https://bmclab.pesquisa.ufabc.edu.br/ensino/biomecanica-i/)**  
         - **[Biomecânica II](https://bmclab.pesquisa.ufabc.edu.br/ensino/biomecanica-ii/)**  
         - **[Modelagem e simulação do movimento humano](https://github.com/BMClab/BMC/blob/master/courses/ModSim2019.md)** (Biomecãnica III)

        The book [*Introduction to Statics and Dynamics*](http://ruina.tam.cornell.edu/Book/index.html), written by Andy Ruina and Rudra Pratap, is an excellent reference (a rigorous and yet didactic presentation of Mechanics for undergraduate students) on Classical Mechanics and the authors kindly offer the book freely available online.  
        We will use this book as the main reference on Mechanics and Mathematics in our course.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

         - [Biomechanics @ Wikipedia](https://en.wikipedia.org/wiki/Biomechanics)  
         - [Latest papers on Biomechanics @ Nature](https://www.nature.com/subjects/biomechanics)  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the Internet

         - [Engineering Your Future - Biomechanical Engineer](https://www.youtube.com/watch?v=aWUavsk2djI)  
         - [What is Biomechanical Engineering?](https://www.youtube.com/watch?v=CXdY0GPRHXo)  
         - [The Weird World of Eadweard Muybridge](https://www.youtube.com/watch?v=5Awo-P3t4Ho) - a video about [Eadweard Muybridge](https://en.wikipedia.org/wiki/Eadweard_Muybridge), an important person to the development of instrumentation for biomechanics.
         - [A complete course on Biomechanics of Movement](https://www.youtube.com/watch?v=VbUNOFgYcKI&list=PL_uk_kfAmFLrtzEfv6njXooOPae3jI1q6)  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. Find examples of applications of biomechanics in different areas.  
        2. Think about practical problems in nature that can be studied in biomechanics with simple approaches (simple modeling and low-tech methods) or very complicated approaches (complex modeling and high-tech methods).  
        3. What the study in the biomechanics of athletes, children, elderlies, persons with disabilities, other animals, and computer animation for the cinema industry may have in common and different?  
        4. Visit the website of the Laboratory of Biomechanics and Motor Control at UFABC and find out what we do and if there is anything you are interested in.  
        5. Is there anything in biomechanics that interests you? How could you pursue this interest?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - [Biomechanics - Wikipedia, the free encyclopedia](http://en.wikipedia.org/wiki/Biomechanics)
        - [Mechanics - Wikipedia, the free encyclopedia](http://en.wikipedia.org/wiki/Mechanics)
        - [International Society of Biomechanics](http://isbweb.org/)
        - [Biomech-l, the biomechanics' e-mail list](http://biomch-l.isbweb.org/)
        - [Journal of Biomechanics' aims](http://www.jbiomech.com/aims)  
        - <a href="http://courses.washington.edu/bioen520/notes/History_of_Biomechanics_(Martin_1999).pdf">A Genealogy of Biomechanics</a>
        - Fung Y-C (1993) [Biomechanics: mechanical properties of living tissues](https://books.google.com.br/books?id=yx3aBwAAQBAJ). 2nd ed. Springer.  
        - Hatze H (1974) [The meaning of the term biomechanics](https://github.com/demotu/BMC/blob/master/courses/refs/HatzeJB74biomechanics.pdf). Journal of Biomechanics, 7, 189–190.  
        - Hibbeler RC (2012) [Engineering Mechanics: Statics](http://books.google.com.br/books?id=PSEvAAAAQBAJ). 13 edition. Prentice Hall.   
        - Nigg BM and Herzog W (2006) [Biomechanics of the Musculo-skeletal System](https://books.google.com.br/books?id=hOIeAQAAIAAJ&dq=editions:ISBN0470017678). 3rd Edition. Wiley.  
        - Ruina A, Rudra P (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.  
        - Winter DA (2009) [Biomechanics and motor control of human movement](http://books.google.com.br/books?id=_bFHL08IWfwC). 4 ed. Hoboken: Wiley.
        - Zatsiorsky VM (1997) [Kinematics of Human Motion](http://books.google.com.br/books/about/Kinematics_of_Human_Motion.html?id=Pql_xXdbrMcC&redir_esc=y). Champaign, Human Kinetics.  
        - Zatsiorsky VM (2002) [Kinetics of human motion](http://books.google.com.br/books?id=wp3zt7oF8a0C). Human Kinetics.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
