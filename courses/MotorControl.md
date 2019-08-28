# Introduction to modeling and simulation of human movement

### Instructor  
- [Marcos Duarte](http://pesquisa.ufabc.edu.br/bmclab/people/marcos-duarte/)
- [Renato Watanabe](pesquisa.ufabc.edu.br/bmclab/pessoal/renato/)

### Time and place  
- Wednesdays, 19-23h; O-L07 campus São Bernardo do Campo

### Grades

* [Grades](https://docs.google.com/spreadsheets/d/e/2PACX-1vS0vCERbEbcBLo_4fjZh5pl4i6-6Bk8fRCTnqWc0LI0CzTnXq5wSYSz5ojaG5Uda0mSf5xL6k0Ml06c/pubhtml)

## Lecture Schedule

### Lecture 1

[Introduction](MotorControl2019_0.pdf)

 * Control of human movement, modeling and simulation  
 * Course Information   
 * [OpenSim software](https://simtk.org/projects/opensim), [Mendeley](https://www.mendeley.com), [Github](https://www.github.com), [Python](https://www.python.org/), [Anaconda](https://www.anaconda.com/) 


 * [Python for scientific computing](http://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/PythonForScientificComputing.ipynb)
 * [Python Tutorial](https://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/PythonTutorial.ipynb)
 
Tasks (for Lecture 2):

*  Install OpenSim and do the first three tutorials (Help menu).

- Write a Jupyter notebook to compute and plot the acceleration of the particle with displacement data contained in the file [pezzack.txt](http://isbweb.org/data/pezzack/index.html). Additionally, compare the computed acceleration with the true acceleration (contained in the same data).
 


 Readings (for Lecture 3):

* Bear, M., Connors, B., & Paradiso, M. (2017). Neuroscience: Exploring the brain (4th ed.). (pages 454-468).

 Readings:

* Shen, H. (2014). Interactive notebooks: Sharing the code. Nature, 5–6. Retrieved from http://europepmc.org/abstract/med/25373681  
* Perkel, J. M. (2015). Programming: Pick up Python. Nature, 518(7537), 125–126. https://doi.org/10.1038/518125a
 

### Lecture 2

* OpenSim tutorials

* Numerical methods to solve ordinary differential equations. [https://github.com/BMClab/bmc/blob/master/notebooks/newtonLawForParticles.ipynb](https://github.com/BMClab/bmc/blob/master/notebooks/newtonLawForParticles.ipynb),[https://github.com/BMClab/bmc/blob/master/notebooks/OrdinaryDifferentialEquation.ipynb](https://github.com/BMClab/bmc/blob/master/notebooks/OrdinaryDifferentialEquation.ipynb)

 
 Task for lecture 3:
 
 * Write a Jupyter notebook to find the trajectory of a ball considering the air drag proportional to the square root of the ball velocity. Consider that the initial velocity of the particle is $v_0 = 30 m/s$ and the initial angle with the horizontal plane is 30 degrees, the gravity acceleration is $g = 9.81 m/s^2$, the mass of the ball is m = 0.43 kg and the air drag coefficient is $b= 0.006 kg.m^{1/2}/s^{3/2}$.

Readings (for Lecture 3):

* Bear, M., Connors, B., & Paradiso, M. (2017). Neuroscience: Exploring the brain (4th ed.). (pages 454-468).

### Lecture 3

 * Discussion about Bear, M., Connors, B., & Paradiso, M. (2017). Neuroscience: Exploring the brain (4th ed.). (pages 454-468) 
   + Animations about muscle contraction: [1](https://youtu.be/GrHsiHazpsw), [2](https://youtu.be/jqy0i1KXUO4), [3](https://www.youtube.com/watch?v=m0UiYgnWaU8)  

* Readings (for Lecture 4):
 
    + NIGG, B. M.; HERZOG, W. Modelling. In: Biomechanics of the Musculo-skeletal System.  p. 501 - 534.

 * Models of Kelvin, Voigt and Maxwell.    
   
  Tasks (for lecture 4)
  
  * Implement the Kelvin model. Set the parameters of the model so as to the response of the model is similar to the shown in the Figure 3(a) from VAN LOOCKE, M.; LYONS, C. G.; SIMMS, C. K. Viscoelastic properties of passive skeletal muscle in compression: Stress-relaxation behaviour and constitutive modelling. Journal of Biomechanics, v. 41, n. 7, p. 1555–1566, 2008. You can find this article at the Mendeley group.

  Additional reading:
  
  * YAMAGUCHI, Y. T; Dynamic modeling of musculoskeletal motion: A Vectorized Approach for Biomechanical Analysis in Three Dimensions (2001), Sections 2.2.1 and 2.2.2.
  
### Lecture 4

* Discussion about NIGG, B. M.; HERZOG, W. Modelling. In: Biomechanics of the Musculo-skeletal System.  p. 501 - 534.

Task (for Lecture 5)

- Implement the model in NIGG, B. M.; HERZOG, W. Modelling. In: Biomechanics of the Musculo-skeletal System.  p. 631 - 634. 
    - Use Fmax = 7400 N, Lceopt = 0.093 m, W = 0.63Lceopt, a = 0.25Fmax, b = 2.5Lceopt, LslackSE = 0.223 m, LslackPE = 0.093 m, kSE = Fmax/(0.04LslackSE)^2, kPE = Fmax/(0.04LslackPE)^2.
    - The initial condition of Lce is Lce = 0.087 m.
    - The length of muscle-tendon should be the shown in Figure 4.8.6 of   NIGG, B. M.; HERZOG, W. Modelling. In: Biomechanics of the Musculo-skeletal System. 

### Lecture 5

  - Implementation of the model in NIGG, B. M.; HERZOG, W. Modelling. In: Biomechanics of the Musculo-skeletal System.  p. 631 - 634.

  Task (for Lecture 6)
  
  * Change the derivative of the contractile element length function. The new function must compute the derivative according to the article from Thelen (2003) (Eq. (4), (6) and (7)):
  
    + Thelen D; Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older adults (2003)

  -[Nigg muscle model](https://gist.github.com/rnwatanabe/1f6fb066fda122d067bcd91c4a8f082b)

  ### Lecture 6


Task (for Lecture 7):
 
 * Implement the activation dynamics as in Thelen (2003).
 
 * Implement the pennation angle as in Thelen (2003).

 ### Lecture 7

* Task (for Lecture 8):

- Define functions to compute all variables of the muscle model.

* [Tutorial on classes in Python ](https://panda.ime.usp.br/pensepy/static/pensepy/13-Classes/classesintro.html)  

Task (for Lecture 8):
 
 * Write a Python Class to implement the muscle model developed during the course. [Continue from here](https://gist.github.com/rnwatanabe/8698b381302eff7d481398cad9407ef7)

Readings (for Lecture 8)
  
   * BEAR, M. F.; W, C. B.; PARADISO, M. A. Neuroscience: Exploring the brain. 4. ed. Philadelphia, PA, USA: Lippincott Williams & Wilkins, 2015.   p. 469-481.

### Lecture 8

- Task(for Lecture 9):

Implement the knee joint using the msucle implemented during the course: 

The data are:

- Lmt = 0.31 - (\theta  - \pi/2)Rf
-Rf = 0.033 m
-Rcm = 0.264 m
- m = 10 kg
- I = 0.1832 kgm^2

The knee angle dynamics must be modelled according to the Newton-Euler laws.