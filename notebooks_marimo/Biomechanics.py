import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to Biomechanics

    > Marcos Duarte, Renato Naville Watanabe,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    This notebook is the front door to a collection of notes on biomechanics and scientific computing. Unlike the other notebooks here, it does not teach a method. It answers three questions: what biomechanics is, how it came to be that way, and where in this collection you should go next.

    Read it once from top to bottom. Then come back to the section **Where to go from here**, which lays out study paths through the rest of the material; that section is meant to be revisited, not memorized.

    There is one calculation in the middle. Do not skip it. It is small, but it is the shortest honest demonstration of why biomechanics needs mechanics at all, and it reproduces a result from the 1680s that still surprises people.

    **Challenge 0.** Before reading further, write down one movement you would like to understand: something from your sport, from a clinic, from a workplace, or from an animal you have watched. Keep it on your scratchpad. Every section below will ask you to look at it again.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is biomechanics?

    The etymology of the word is transparent:

    $$\text{Biomechanics} := \text{bios} \, (\text{life}) + \text{mechanics}$$

    Professor Herbert Hatze, in a 1974 letter to the editors of the Journal of Biomechanics, proposed a definition that has held up remarkably well:

    > "*Biomechanics is the study of the structure and function of biological systems by means of the methods of mechanics.*"
    >
    > Hatze (1974) [The meaning of the term biomechanics](https://github.com/demotu/BMC/blob/master/courses/refs/HatzeJB74biomechanics.pdf).

    Read that definition slowly, because every phrase is doing work. *Structure and function* — not only how a body moves, but how it is built and what it is for. *Biological systems* — not only humans, and not only whole bodies; a tendon, a cell membrane, a tree, and a swarm of insects all qualify. *By means of the methods of mechanics* — this is the clause that separates biomechanics from the rest of biology, and it is also the clause that will occupy most of your study time.

    **Guiding questions 1.**

    1. Using Hatze's definition, is the movement you chose in Challenge 0 a biomechanics problem? Which part of it is *structure* and which part is *function*?
    2. What would you have to measure to study it?
    3. Which of those measurements do you think already exists as a standard laboratory method?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is biomechanics a science of its own?

    Hatze thought so, and he was emphatic that biomechanics is **not** simply mechanics applied to living things:

    > "*It would not be correct to state that 'Biomechanics is the study of the mechanical aspects of the structure and function of biological systems' because biological systems do not have mechanical aspects. They only have biomechanical aspects (otherwise mechanics, as it exists, would be sufficient to describe all phenomena which we now call biomechanical features of biological systems).*" Hatze (1974)

    His illustration is a javelin throw. Once the javelin leaves the thrower's hand, its flight is a problem in aerodynamics and ballistics: existing mechanical models are sufficient, and nothing about the problem is biological. The phase *before* release is different. There is no off-the-shelf mechanical model for a human body accelerating a shaft through a run-up, a plant, and a whip-like sequence of joint rotations. That phase, Hatze argues, is where biomechanics begins.

    Professor Yuan-Cheng Fung, another founder of the field, was far less enthusiastic about the distinction:

    > "*Biomechanics is mechanics applied to biology*" Fung (1993).

    This is not a demotion of the field. A few pages later in the same book, Fung writes:

    > "*Biomechanics has participated in virtually every modern advance of medical science and technology*" Fung (1993).

    The disagreement is worth sitting with rather than resolving. Hatze is pointing at the problems for which no adequate mechanical model exists yet; Fung is pointing at the enormous amount of useful work that gets done with the mechanics we already have. In practice you will do both, and it helps to know which one you are doing at any given moment.

    **Challenge 1.** Take the movement from Challenge 0 and split it into a phase where textbook mechanics would be sufficient and a phase where it would not. Which phase interests you more? Which is harder to measure?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where the word gets stretched

    Hatze saw a terminological problem coming, already in 1974:

    > "*The use of the term biomechanics imposes rather severe restrictions on its meaning because of the established definition of the term, mechanics. This is unfortunate, since the synonym Biomechanics, as it is being understood by the majority of biomechanists today, has a much wider meaning.*" Hatze (1974)

    He was right, and the stretch has only grown. It is common to see methods filed under biomechanics that have no relation to mechanics at all. Electromyography and thermography are two examples: the former is genuinely useful in biomechanics and appears later in this collection, but electromyography measures an electrical phenomenon, and thermography a thermal one. Electromagnetism and thermodynamics are [other branches of physics](https://en.wikipedia.org/wiki/Branches_of_physics), even though mechanics and thermodynamics overlap considerably.

    None of this makes such methods less valuable. It just means that when someone says "we did a biomechanical analysis", the useful follow-up question is: *which mechanical quantity did you actually measure or compute?*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Biomechanics and engineering

    Even taking Hatze's view that biomechanics is a science in its own right, it is entirely reasonable to speak of *engineering biomechanics*, because:

    - *Engineering is the application of scientific and mathematical principles to practical ends* [[The Free Dictionary](http://www.thefreedictionary.com/engineering)];
    - *Engineering Mechanics is the application of Mechanics to solve problems involving common engineering elements* [[Wikibooks](https://en.wikibooks.org/wiki/Engineering_Mechanics)];
    - *Biomedical engineering is the application of engineering principles and design concepts to medicine and biology for healthcare purposes* [[Wikipedia](https://en.wikipedia.org/wiki/Biomedical_engineering)].

    Biomechanics matters to every field of science and technology touching biology and health, and it is also a source of ideas for synthetic systems inspired by biological ones, as in robotics. For a sense of how wide the range of applications is, browse the current scope of the [Journal of Biomechanics](https://www.sciencedirect.com/journal/journal-of-biomechanics/about/aims-and-scope) — the list of topics runs from molecular and cellular mechanics to whole-body movement and prosthetics. The [Wikipedia article on biomechanics](https://en.wikipedia.org/wiki/Biomechanics) is another good place to get oriented.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The mechanics you actually need

    **A good knowledge of mechanics is a necessary condition, but not a sufficient one, for a good knowledge of biomechanics.**

    The good news is that only a subset of mechanics is relevant. The bodies studied in biomechanics move at moderate speeds $(\ll 3 \times 10^{8}\,\mathrm{m/s})$ and are not very small $(\gg 3 \times 10^{-9}\,\mathrm{m})$, which places them squarely in classical mechanics, as shown in the diagram below. You will not need relativity or quantum mechanics to analyze a gait cycle.

    <figure><center><img src="http://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Physicsdomains.svg/500px-Physicsdomains.svg.png" width=500 alt="Domains of mechanics"/></center><figcaption><center><i>Figure. Domains of mechanics (image from <a href="http://en.wikipedia.org/wiki/Classical_mechanics">http://en.wikipedia.org/wiki/Classical_mechanics</a>).</i></center></figcaption></figure>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How mechanics is usually divided

    Mechanics is the branch of the physical sciences concerned with the state of rest or motion of bodies subjected to the action of forces. It is usually subdivided into three branches according to how the body is idealized: rigid-body mechanics, deformable-body mechanics, and fluid mechanics (Hibbeler, 2012; Ruina and Pratap, 2019).

    Within that, classical mechanics is typically partitioned first into **statics** and **dynamics**, and dynamics is then divided into **kinematics** and **kinetics**. The logic is clean: statics studies forces in the absence of changes in motion, dynamics studies bodies whose motion changes, kinematics describes motion without asking what caused it, and kinetics studies the causes.

    <figure><center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Mechanics_Overview_Table.jpg/640px-Mechanics_Overview_Table.jpg" width=800 alt="Branches of mechanics"/></center><figcaption><center><i>Figure. Branches of mechanics (image from <a href="https://en.wikibooks.org/wiki/Engineering_Statics/Introduction">https://en.wikibooks.org/wiki/Engineering_Statics/Introduction</a>).</i></center></figcaption></figure>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How biomechanics divides it differently

    Now a warning, because this trips up people who arrive from an engineering course.

    It is common in biomechanics to use a different order: to partition the field first into kinematics and kinetics, and only then to divide kinetics into statics and dynamics. David Winter, Nigg and Herzog, and Vladimir Zatsiorsky, among others, organize their books this way.

    The rationale is practical. The first decision a movement scientist makes is whether to consider the causes of motion at all, because it is far simpler to measure and describe the kinematics of human motion than to get at the forces behind it. You can film someone walking with a phone; measuring the force in their soleus tendon is a different kind of undertaking. Ordering the field by that difficulty is useful when you are planning a study.

    These two classifications genuinely conflict, and some scholars will tell you the biomechanics taxonomy is simply wrong and should be aligned with mechanics. Be aware of both, and notice which one a book or a paper is using before you assume you disagree with it.

    **Challenge 2.** Which of the two classifications would put *your* movement from Challenge 0 into the easier category to study first? Does that match your intuition about where you would start?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A worked example: why muscles pull so hard

    Here is the smallest calculation that shows why biomechanics needs mechanics.

    Hold a 5 kg dumbbell in your hand with your forearm horizontal and your upper arm vertical. Nothing is moving, so this is a statics problem — the simplest kind there is.

    The elbow is the pivot. Three things create a moment about it: the weight of the load in your hand, the weight of your own forearm and hand, and the pull of the elbow flexor muscles. The load sits roughly $0.30\,\mathrm{m}$ from the elbow. Your forearm and hand weigh about $1.5\,\mathrm{kg}$, with their center of mass around $0.12\,\mathrm{m}$ from the elbow. The biceps, however, inserts on the radius only about $0.04\,\mathrm{m}$ from the joint.

    That last number is the whole story. Static equilibrium requires the moments about the elbow to sum to zero:

    $$F_m \, d_m = W_{load} \, d_{load} + W_{forearm} \, d_{forearm}$$

    so the muscle force is

    $$F_m = \frac{W_{load} \, d_{load} + W_{forearm} \, d_{forearm}}{d_m}$$

    **Before you run the next cells**, predict $F_m$. Is it smaller than the weight of the load, about the same, or larger? If larger, by roughly what factor? Write your guess down.
    """)
    return


@app.function
def elbow_muscle_force(
    load_mass=5.0,
    muscle_moment_arm=0.04,
    load_distance=0.30,
    forearm_mass=1.5,
    forearm_distance=0.12,
    gravity=9.81,
):
    """Static equilibrium of an elbow holding a load with the forearm horizontal.

    All distances are measured from the elbow joint, in meters, and all
    masses in kilograms.

    Returns the elbow flexor force [N], the elbow joint reaction force [N],
    and the ratio between the muscle force and the weight of the load.
    """
    load_weight = load_mass * gravity
    forearm_weight = forearm_mass * gravity
    moment = load_weight * load_distance + forearm_weight * forearm_distance
    muscle_force = moment / muscle_moment_arm
    joint_force = muscle_force - load_weight - forearm_weight
    force_ratio = muscle_force / load_weight if load_weight > 0 else float("nan")
    return muscle_force, joint_force, force_ratio


@app.cell
def _():
    muscle_force, joint_force, force_ratio = elbow_muscle_force()

    print(f"Weight of the 5 kg load:     {5.0 * 9.81:6.0f} N")
    print(f"Elbow flexor force:          {muscle_force:6.0f} N")
    print(f"Elbow joint reaction force:  {joint_force:6.0f} N")
    print(f"Muscle force / load weight:  {force_ratio:6.1f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The muscle pulls with roughly eight times the weight of the load it is holding, and the elbow joint is compressed by a force several times larger than the load as well. Almost nobody guesses this on the first try.

    This is not a design flaw. It is the price of a lever built for speed. The same ratio $d_{load}/d_m = 0.30/0.04 = 7.5$ that multiplies the required force also means that shortening the muscle by $1\,\mathrm{cm}$ sweeps the hand through $7.5\,\mathrm{cm}$ — and does so $7.5$ times faster than the muscle itself contracts. The musculoskeletal system trades force for range and speed of motion.

    You have just reproduced, with a moment balance, the insight that Giovanni Alfonso Borelli published in the 1680s — before Newton published the laws of motion. We will meet him again in the next section.

    The plot below shows how sensitive this is: the muscle force as a function of where the muscle inserts, and as a function of how heavy the load is.
    """)
    return


@app.function
def plot_elbow_lever():
    """Plot the elbow flexor force against moment arm and against load mass."""
    import matplotlib.pyplot as plt
    import numpy as np

    moment_arms = np.linspace(0.02, 0.08, 200)
    force_vs_arm = np.array(
        [elbow_muscle_force(muscle_moment_arm=d)[0] for d in moment_arms]
    )

    load_masses = np.linspace(0.0, 20.0, 200)
    force_vs_load = np.array([elbow_muscle_force(load_mass=m)[0] for m in load_masses])

    reference_force = elbow_muscle_force()[0]

    _, axs = plt.subplots(1, 2, figsize=(11, 4))

    axs[0].plot(100 * moment_arms, force_vs_arm, "b", linewidth=2)
    axs[0].plot(4.0, reference_force, "ro", markersize=9, label="biceps, 5 kg load")
    axs[0].set_xlabel("Muscle moment arm [cm]")
    axs[0].set_ylabel("Elbow flexor force [N]")
    axs[0].set_title("Where the muscle inserts")
    axs[0].legend()

    axs[1].plot(load_masses, force_vs_load, "b", linewidth=2, label="Muscle force")
    axs[1].plot(
        load_masses,
        9.81 * load_masses,
        "g--",
        linewidth=2,
        label="Weight of the load",
    )
    axs[1].plot(5.0, reference_force, "ro", markersize=9)
    axs[1].set_xlabel("Load mass [kg]")
    axs[1].set_ylabel("Force [N]")
    axs[1].set_title("How heavy the load is")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


@app.cell
def _():
    plot_elbow_lever()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Challenge 3.** Work through these with the function above.

    1. A surgeon moves a tendon insertion from $0.04\,\mathrm{m}$ to $0.05\,\mathrm{m}$ from the joint. By what percentage does the required muscle force drop? What does the patient lose in exchange?
    2. Keep the load at 5 kg but hold it close to your body, at $0.15\,\mathrm{m}$ from the elbow instead of $0.30\,\mathrm{m}$. Explain the result to someone lifting a heavy box.
    3. The joint reaction force grows almost as fast as the muscle force. Why would that matter to someone studying cartilage wear, or designing an elbow prosthesis?
    4. This model ignored the fact that several muscles cross the elbow. If three flexors share the job, can you determine how much force each one produces from statics alone? (This question has a name — muscle redundancy — and it is why [optimization](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/Optimization.ipynb) appears later in this collection.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A short history

    Biomechanics advanced on two legs: progress in mechanics itself, and the invention of instruments for measuring mechanical quantities and for computing with the measurements. Neither leg alone gets you very far. Borelli had a correct idea about levers and no way to measure joint forces; a modern laboratory has superb instruments and still needs a model to interpret what they record.

    The field also required people to become interested in the structure and function of biological systems for their own sake. So it is natural to count the great minds of mechanics — Galileo, Newton, Euler, Lagrange — as contributors to biomechanics, even though none of them set out to found it.

    What follows is a very short walk through some direct contributors. Watch for one thread in particular: the slow arrival of *measurement* as the arbiter of what is true.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Aristotle (384-322 BC)

    Aristotle was the first to write about the movement of animals, in *On the Motion of Animals (De Motu Animalium)* and *On the Gait of Animals (De Incessu Animalium)* [[Works by Aristotle](http://classics.mit.edu/Browse/index-Aristotle.html)].

    He already knew what we now call Newton's third law of motion:

    > "*For as the pusher pushes so is the pushed pushed, and with equal force.*" [Part 3, [On the Motion of Animals](http://classics.mit.edu/Aristotle/motion_animals.html)]

    That is a remarkable sentence to find two thousand years early. But the rest of his physics shows what was missing. Aristotle does not appear to have used anything resembling the [scientific method](https://en.wikipedia.org/wiki/Scientific_method) — systematic observation, measurement, and experiment, with hypotheses formulated, tested, and revised.

    Much of his physics was therefore ambiguous or wrong. For him there was no motion without a force, and he deduced that speed was proportional to force and inversely proportional to resistance [[Book VII, Physics](http://classics.mit.edu/Aristotle/physics.7.vii.html)]. That conclusion is not unreasonable if the motions you observe are all dominated by friction, which they were.

    How little he checked is captured, ironically, in one of his own lines:

    > "Males have more teeth than females in the case of men, sheep, goats, and swine; in the case of other animals observations have not yet been made." Aristotle, [The History of Animals](http://classics.mit.edu/Aristotle/history_anim.html).

    Counting teeth would have taken an afternoon.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Leonardo da Vinci (1452-1519)

    <div><figure><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Da_Vinci_Vitruve_Luc_Viatour.jpg/353px-Da_Vinci_Vitruve_Luc_Viatour.jpg' width="240" alt="Vitruvian Man" style="float:right;margin: 0 0 0 20px;"/></figure></div>

    Leonardo contributed to biomechanics by looking, carefully and repeatedly:

    - studies on the proportions of humans and animals;
    - anatomy studies of the human body, especially the foot;
    - studies on the mechanical function of muscles.

    The shift from Aristotle is the shift from reasoning about bodies to dissecting and measuring them.

    <br><br>
    Figure. *"Le proporzioni del corpo umano secondo Vitruvio", also known as the [Vitruvian Man](https://en.wikipedia.org/wiki/Vitruvian_Man), drawing by [Leonardo da Vinci](https://en.wikipedia.org/wiki/Leonardo_da_Vinci) circa 1490 based on the work of [Marcus Vitruvius Pollio](https://en.wikipedia.org/wiki/Vitruvius) (1st century BC), depicting a man in supposedly ideal human proportions (image from [Wikipedia](https://en.wikipedia.org/wiki/Vitruvian_Man)).*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Giovanni Alfonso Borelli (1608-1679)

    <div><figure><img src='https://upload.wikimedia.org/wikipedia/commons/d/d5/Giovanni_Borelli_-_lim_joints_%28De_Motu_Animalium%29.jpg' width="240" alt="Borelli" style="float:right;margin: 0 0 0 20px;"/></figure></div>

    Borelli is [the father of biomechanics](https://en.wikipedia.org/wiki/Giovanni_Alfonso_Borelli), and the first to bring the modern scientific method to the subject, in his book [De Motu Animalium](http://www.e-rara.ch/doi/10.3931/e-rara-28707):

    - he proposed that the levers of the musculoskeletal system magnify motion rather than force — the result you computed above;
    - he calculated the forces required for equilibrium at various joints of the human body, *before* Newton published the laws of motion.

    <br><br>
    Figure. *Excerpt from the book De Motu Animalium*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Eadweard James Muybridge (1830-1904) and Étienne-Jules Marey (1830-1904)

    If Borelli supplied the models, these two supplied the instruments — and with them, the other leg the field had been missing:

    - [Eadweard Muybridge](https://en.wikipedia.org/wiki/Eadweard_Muybridge): development of photography for movement analysis.
    - [Étienne-Jules Marey](https://en.wikipedia.org/wiki/%C3%89tienne-Jules_Marey): development of the first devices for measuring foot-ground contact forces.

    Every motion-capture system and every force plate in use today is a descendant of this work. When you reach the notebooks on kinematics and on force plates later in this collection, you are using their instruments with better sensors.

    <center>
    <div><figure><img src='https://upload.wikimedia.org/wikipedia/commons/d/d2/The_Horse_in_Motion_high_res.jpg' width="400" alt="The Horse in Motion, by Muybridge"/></figure></div>
    <div><figure><img src='https://upload.wikimedia.org/wikipedia/commons/2/2d/Animal_mechanism-_a_treatise_on_terrestrial_and_a%C3%ABrial_locomotion_%281874%29_%2818198040265%29.jpg' width="400" alt="Animal mechanism, by Marey"/></figure></div>
    </center>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### More on the history of biomechanics

    - [A Genealogy of Biomechanics](https://courses.washington.edu/bioen520/notes/History_of_Biomechanics_(Martin_1999).pdf)
    - [History of Biomechanics and Kinesiology](https://biomechanics.vtheatre.net/doc/history.html)
    - Chapter 1 of Nigg and Herzog (2006) [Biomechanics of the Musculo-skeletal System](https://books.google.com.br/books?id=hOIeAQAAIAAJ&dq=editions:ISBN0470017678)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Biomechanics today

    The biomechanics community has an official scientific society, the [International Society of Biomechanics](http://isbweb.org/), a journal, the [Journal of Biomechanics](http://www.jbiomech.com), and a long-running mailing list, [Biomch-L](http://biomch-l.isbweb.org). If you are serious about the field, subscribing to the list is a cheap way to see what practitioners actually argue about.

    For a concrete sense of the work, here are two examples from our own laboratory:

    - Clinical gait analysis: [https://bmclab.pesquisa.ufabc.edu.br/servicos/cga/](https://bmclab.pesquisa.ufabc.edu.br/servicos/cga/)
    - Biomechanics of sports: [https://bmclab.pesquisa.ufabc.edu.br/biomechanics-of-the-bicycle-kick/](https://bmclab.pesquisa.ufabc.edu.br/biomechanics-of-the-bicycle-kick/)

    ### The future of the field

    (Human) Movement Science combines many disciplines — physiology, biomechanics, psychology — for the study of human movement. Professor Benno Nigg argued that as concern for human well-being grows, movement science will take on an increasingly central role:

    > Movement science will be one of the most important and most recognized science fields in the twenty-first century... The future discipline of movement science has a unique opportunity to become an important contributor to the well-being of mankind.<br>
    > Nigg BM (1993) [Sport science in the twenty-first century](http://www.ncbi.nlm.nih.gov/pubmed/8230394). Journal of Sports Sciences, 77, 343-347.

    Biomechanics will contribute to that in proportion to how well its practitioners can measure, model, and compute. Which brings us to the rest of this collection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where to go from here

    The notebooks in this repository are notes on scientific computing and data analysis for biomechanics and motor control. There are many of them, and reading them in alphabetical order would be a poor idea. Below are six study paths. Each one exists for a reason, stated first; the notebooks listed are the backbone of that path, not an exhaustive list.

    The complete index lives in the repository [README](https://github.com/BMClab/BMC/blob/master/README.md). The `notebooks_marimo/` directory holds the marimo versions, which are the ones under active development.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Track 1 — Tools of the trade

    You cannot analyze movement you cannot load, plot, or share. This track is not about biomechanics; it is about not being blocked by your own tooling. Take it first if you are new to scientific Python, and skim it otherwise.

    - [Python for scientific computing](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/PythonForScientificComputing.ipynb)
    - [Python tutorial](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/PythonTutorial.ipynb)
    - [Code structure for data analysis](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/CodeStructure.ipynb)
    - [Version control with Git and GitHub](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/VersionControlGitGitHub.ipynb)

    ### Track 2 — From a raw recording to a usable signal

    Every measurement in biomechanics arrives contaminated: sensor noise, missing markers, trials of different durations, subjects of different sizes. Almost every mistake in an analysis is made here, before any mechanics is applied. This track teaches you to get from a file to a signal you can trust.

    - [Basic properties of signals](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/SignalBasicProperties.ipynb)
    - [Data filtering](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb)
    - [Residual analysis for the optimal cutoff frequency](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb)
    - [Time normalization of data](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/TimeNormalization.ipynb)
    - [Ensemble average](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/EnsembleAverage.ipynb)
    - [Open files in C3D format](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/OpenC3Dfile.ipynb)

    ### Track 3 — Kinematics: describing motion

    This is the easier half of the biomechanics partition, and the right place to start doing actual mechanics. Everything here answers *how did it move*, without yet asking why. Note how much of the work is really about choosing a frame of reference and sticking to it.

    - [Frame of reference](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb)
    - [Kinematics of a particle: one-dimensional motion](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb)
    - [Projectile motion](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/ProjectileMotion.ipynb)
    - [Angular kinematics (2D)](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/KinematicsAngular2D.ipynb)
    - [Rigid-body transformations (2D)](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/Transformation2D.ipynb) and [(3D)](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/Transformation3D.ipynb)
    - [Kinematic chain](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/KinematicChain.ipynb)

    ### Track 4 — Kinetics: the forces behind the motion

    This track generalizes the elbow calculation you did above: from one joint in static equilibrium to whole limbs in motion. The free-body diagram is the single most useful habit in the entire field — if you learn nothing else from this repository, learn to draw one properly.

    - [Fundamental concepts of kinetics](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/KineticsFundamentalConcepts.ipynb)
    - [Center of mass and moment of inertia](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/CenterOfMassAndMomentOfInertia.ipynb)
    - [Newton's laws for particles](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/newtonLawForParticles.ipynb)
    - [Free body diagram](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/FreeBodyDiagram.ipynb)
    - [Lagrangian mechanics](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/lagrangian_mechanics.ipynb)

    ### Track 5 — Modeling and simulating the musculoskeletal system

    Measurement alone cannot tell you what an individual muscle did; there are more muscles than equations. Modeling and simulation are how biomechanics gets at quantities no instrument can record directly. This is also where Hatze's argument becomes concrete, since no off-the-shelf mechanical model covers a contracting muscle.

    - [Ordinary differential equations](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/OrdinaryDifferentialEquation.ipynb)
    - [Body segment parameters](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/BodySegmentParameters.ipynb)
    - [Muscle modeling](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/MuscleModeling.ipynb) and [muscle simulation](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/MuscleSimulation.ipynb)
    - [Musculoskeletal modeling and simulation](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/MusculoskeletaModelingSimulation.ipynb)
    - [Multibody dynamics of simple biomechanical models](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/MultibodyDynamics.ipynb)
    - [Optimization](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/Optimization.ipynb)

    ### Track 6 — Whole tasks, end to end

    Finally, complete analyses of real movements, each combining every track above. Read one of these early anyway, even before you can follow all the steps: it shows you what the destination looks like.

    - [The inverted pendulum model of human standing posture](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/IP_Model.ipynb)
    - [Measurements in stabilography](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/Stabilography.ipynb)
    - [Biomechanical analysis of vertical jumps](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/VerticalJump.ipynb)
    - [Gait analysis (2D)](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/GaitAnalysis2D.ipynb)
    - [Introduction to data analysis in electromyography](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/Electromyography.ipynb)

    **Challenge 4.** Go back to the movement you chose in Challenge 0. Which single notebook from the list above would help you most right now? Open it next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Biomechanics at UFABC

    At the university level, mechanics is usually taught across separate disciplines: Statics and Dynamics (rigid-body mechanics), Strength of Materials (deformable-body mechanics), and Fluid Mechanics. A thorough study of biomechanics has to cover all three, because biological systems are rigid, deformable, and fluid depending on which question you ask.

    The Biomedical Engineering degree at UFABC covers these topics for biological systems across several courses: Ciência dos Materiais Biocompatíveis, Modelagem e Simulação de Sistemas Biomédicos, Métodos de Elementos Finitos aplicados a Sistemas Biomédicos, Mecânica dos Fluidos, Caracterização de Biomateriais, Sistemas Biológicos, and Biomecânica I, Biomecânica II, and Modelagem e simulação do movimento humano (Biomecânica III).

    How much biology actually appears in each of these varies a great deal, and none of them covers the study of human motion with implications for health, rehabilitation, and sport, except the last. That is precisely why Biomecânica I and II focus on the analysis of human movement:

    - **[Biomecânica I](https://bmclab.pesquisa.ufabc.edu.br/ensino/biomecanica-i/)**
    - **[Biomecânica II](https://bmclab.pesquisa.ufabc.edu.br/ensino/biomecanica-ii/)**
    - **[Modelagem e simulação do movimento humano](https://github.com/BMClab/BMC/blob/master/courses/ModSim2019.md)** (Biomecânica III)

    The book [*Introduction to Statics and Dynamics*](http://ruina.tam.cornell.edu/Book/index.html), by Andy Ruina and Rudra Pratap, is an excellent reference — a rigorous and yet didactic presentation of mechanics for undergraduates — and the authors kindly make it freely available online. It is our main reference on mechanics and mathematics.

    For comparison, here is how a few other universities organize their biomechanics teaching:

    - [Biomechanical Engineering Courses](https://me.stanford.edu/groups/biomechanical-engineering-program/biomechanical-engineering-courses) (Stanford)
    - [BME 366-0-01: Biomechanics of Movement](https://www.mccormick.northwestern.edu/biomedical/academics/courses/descriptions/366.html) (Northwestern)
    - [Biomechanics at MIT](https://ocw.mit.edu/search/?q=biomechanics)
    - [Biomechanics curriculum](https://www.me.washington.edu/students/grad/curriculum/biomechanics) (Washington)
    - [Online Biomechanics Courses](https://edutestlabs.com/online-biomechanics-courses/)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Find examples of applications of biomechanics in areas you had not associated with the field before. How far from human movement can you get and still be doing biomechanics under Hatze's definition?
    2. Think of practical problems in nature that can be studied in biomechanics with a simple approach (simple modeling and low-tech methods) and of problems that demand a complicated one (complex modeling and high-tech methods). What decides which category a problem falls into?
    3. What do studies on the biomechanics of athletes, children, elderly people, people with disabilities, other animals, and computer animation for the film industry have in common, and where do they diverge?
    4. Visit the website of the [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br) at UFABC and find out what we do. Is there anything there you would want to work on?
    5. Return once more to the movement you chose in Challenge 0. Write one paragraph: what you would measure, what you would have to model because you cannot measure it, and which notebook in this collection you would open first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Reading:

    - [Biomechanics @ Wikipedia](https://en.wikipedia.org/wiki/Biomechanics)
    - [Latest papers on Biomechanics @ Nature](https://www.nature.com/subjects/biomechanics)
    - [Aims and scope of the Journal of Biomechanics](https://www.sciencedirect.com/journal/journal-of-biomechanics/about/aims-and-scope)

    Video lectures:

    - [Engineering Your Future - Biomechanical Engineer](https://www.youtube.com/watch?v=aWUavsk2djI)
    - [What is Biomechanical Engineering?](https://www.youtube.com/watch?v=CXdY0GPRHXo)
    - [The Weird World of Eadweard Muybridge](https://www.youtube.com/watch?v=5Awo-P3t4Ho) — on [Eadweard Muybridge](https://en.wikipedia.org/wiki/Eadweard_Muybridge) and his role in the instrumentation of biomechanics.
    - [A complete course on Biomechanics of Movement](https://www.youtube.com/watch?v=VbUNOFgYcKI&list=PL_uk_kfAmFLrtzEfv6njXooOPae3jI1q6)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Biomechanics - Wikipedia, the free encyclopedia](http://en.wikipedia.org/wiki/Biomechanics)
    - [Mechanics - Wikipedia, the free encyclopedia](http://en.wikipedia.org/wiki/Mechanics)
    - [International Society of Biomechanics](http://isbweb.org/)
    - [Biomch-L, the biomechanics' e-mail list](http://biomch-l.isbweb.org/)
    - [Journal of Biomechanics' aims and scope](https://www.sciencedirect.com/journal/journal-of-biomechanics/about/aims-and-scope)
    - <a href="http://courses.washington.edu/bioen520/notes/History_of_Biomechanics_(Martin_1999).pdf">A Genealogy of Biomechanics</a>
    - Fung Y-C (1993) [Biomechanics: mechanical properties of living tissues](https://books.google.com.br/books?id=yx3aBwAAQBAJ). 2nd ed. Springer.
    - Hatze H (1974) [The meaning of the term biomechanics](https://github.com/demotu/BMC/blob/master/courses/refs/HatzeJB74biomechanics.pdf). Journal of Biomechanics, 7, 189-190.
    - Hibbeler RC (2012) [Engineering Mechanics: Statics](http://books.google.com.br/books?id=PSEvAAAAQBAJ). 13th edition. Prentice Hall.
    - Nigg BM (1993) [Sport science in the twenty-first century](http://www.ncbi.nlm.nih.gov/pubmed/8230394). Journal of Sports Sciences, 77, 343-347.
    - Nigg BM and Herzog W (2006) [Biomechanics of the Musculo-skeletal System](https://books.google.com.br/books?id=hOIeAQAAIAAJ&dq=editions:ISBN0470017678). 3rd Edition. Wiley.
    - Ruina A, Pratap R (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.
    - Winter DA (2009) [Biomechanics and motor control of human movement](http://books.google.com.br/books?id=_bFHL08IWfwC). 4th ed. Hoboken: Wiley.
    - Zatsiorsky VM (1997) [Kinematics of Human Motion](http://books.google.com.br/books/about/Kinematics_of_Human_Motion.html?id=Pql_xXdbrMcC&redir_esc=y). Champaign, Human Kinetics.
    - Zatsiorsky VM (2002) [Kinetics of human motion](http://books.google.com.br/books?id=wp3zt7oF8a0C). Human Kinetics.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
