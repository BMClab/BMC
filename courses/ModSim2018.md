# Introduction to modeling and simulation of human movement

### Instructor  
- [Marcos Duarte](http://demotu.org/people/marcos-duarte/)
- [Renato Watanabe](http://demotu.org/pessoal/renato/)

### Time and place  
- Tuesdays, 17-19h; Thursdays, 17-19h, lab A2 L003, campus São Bernardo do Campo

### Grades

* [Grades](https://docs.google.com/spreadsheets/d/e/2PACX-1vS6yAX5ZHkzhnij4b_lklDwWtV-0KrqTlEQv2W_X2b-w1woXNmBR6p0Mq-IcV51gw7y0EeaBnIC5Xf0/pubhtml)

## Lecture Schedule

### Lecture 1

[Introduction](ModSim2018_0.pdf)

 * Control of human movement, modeling and simulation  
 * Course Information   
 * [OpenSim software](https://simtk.org/projects/opensim), [Mendeley](https://www.mendeley.com), [Github](https://www.github.com), [Python](https://www.python.org/), [Anaconda](https://www.anaconda.com/) 

Tasks (for Lecture 3):

*  Install OpenSim and do the first three tutorials (Help menu).

Readings (for Lecture 5):

* TSIANOS, G. A .; LOEB, G. E. Muscle and limb mechanics. Comprehensive Physiology, v. 7, n. 2, p. 429-462, 2017.
* ZAJAC, F. E. Muscle coordination of movement: a perspective. J Biomech, v. 26 Suppl 1, n. SUPPL. 1, p. 109–124, 1993. 

### Lecture 2

 * [Python for scientific computing](http://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/PythonForScientificComputing.ipynb)
 * [Python Tutorial](https://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/PythonTutorial.ipynb)
 
Tasks (for Lecture 4):

- Write a Jupyter notebook to compute and plot the acceleration of the particle with displacement data contained in the file [pezzack.txt](http://isbweb.org/data/pezzack/index.html). Additionally, compare the computed acceleration with the true acceleration (contained in the same data).
 
Readings:

 * Shen, H. (2014). Interactive notebooks: Sharing the code. Nature, 5–6. Retrieved from http://europepmc.org/abstract/med/25373681  
 * Perkel, J. M. (2015). Programming: Pick up Python. Nature, 518(7537), 125–126. https://doi.org/10.1038/518125a
 
### Lecture 3

 * OpenSim tutorials
 
### Lecture 4

 * Python programming  
   + [Python Tutorial](https://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/PythonTutorial.ipynb)
 
 Tasks (for Lecture 6):
 
  * Plot the trajectory of a particle during 10 seconds with the gravity force acting on it (gravity acceleration g = -9.81 m/s^2). Consider that the initial velocity of the particle is v0 = 30 m/s and the initial angle with the horizontal plane is 30 degrees. 
 
 
### Lecture 5

 * Discussion about TSIANOS and LOEB (2017) and Zajac (1993)  
   + Animations about muscle contraction: [1](https://youtu.be/GrHsiHazpsw), [2](https://youtu.be/jqy0i1KXUO4), [3](https://youtu.be/s_TRsf6tJsc)  
  

### Lecture 6
 
 * Numerical methods to solve ordinary differential equations. [https://github.com/BMClab/bmc/blob/master/notebooks/newtonLawForParticles.ipynb](https://github.com/BMClab/bmc/blob/master/notebooks/newtonLawForParticles.ipynb),[https://github.com/BMClab/bmc/blob/master/notebooks/OrdinaryDifferentialEquation.ipynb](https://github.com/BMClab/bmc/blob/master/notebooks/OrdinaryDifferentialEquation.ipynb)
 
 Task for this class:
 
 * Write a Jupyter notebook to find the trajectory of a ball considering the air drag proportional to the square root of the ball velocity. Consider that the initial velocity of the particle is v0 = 30 m/s and the initial angle with the horizontal plane is 30 degrees, the gravity acceleration is g = 9.81 m/s^2, the mass of the ball is m = 0.43 kg and the air drag coefficient is $b= 0.006 kg.m^{1/2}/s^{3/2}$.
 
 Readings (for Lecture 7):
 
 * NIGG, B. M.; HERZOG, W. Modelling. In: Biomechanics of the Musculo-skeletal System.  p. 501 - 534.  
 
### Lecture 7

  * Discussion about NIGG, B. M.; HERZOG, W. Modelling. In: Biomechanics of the Musculo-skeletal System.  p. 501 - 534.
  
### Lecture 8

  * Models of Kelvin, Voigt and Maxwell.   
  
  Additional reading:
  
  * YAMAGUCHI, Y. T; Dynamic modeling of musculoskeletal motion: A Vectorized Approach for Biomechanical Analysis in Three Dimensions (2001), Sections 2.2.1 and 2.2.2.
  
  Readings (for Lecture 9):
  
  * NIGG, B. M.; HERZOG, W. Modelling. In: Biomechanics of the Musculo-skeletal System.  p. 631 - 634.  
  
  Tasks (for lecture 9)
  
  * Implement the Kelvin model. Set the parameters of the model so as to the response of the model is similar to the shown in the Figure 3(a) from VAN LOOCKE, M.; LYONS, C. G.; SIMMS, C. K. Viscoelastic properties of passive skeletal muscle in compression: Stress-relaxation behaviour and constitutive modelling. Journal of Biomechanics, v. 41, n. 7, p. 1555–1566, 2008. You can find this article at the Mendeley group.
  
  In this study the initial length of the fibre is 10 mm. Then the length decreases to 7 mm with different velocities. 
  
### Lecture 9
 
  * Implement muscle model of Nigg and Herzog.
  
  Readings (for Lecture 11)
  
   * BEAR, M. F.; W, C. B.; PARADISO, M. A. Neuroscience: Exploring the brain. 4. ed. Philadelphia, PA, USA: Lippincott Williams & Wilkins, 2015.  (Chapter 13)
   
### Lecture 10
  
  [Nigg and Herzog model implemented in the class](https://github.com/BMClab/bmc/blob/master/courses/modsim2018/renatowatanabe/Lecture9_MotorControl-Copy1.ipynb)
  
  Task (for Lecture 11)
  
  * Change the derivative of the contractile element length function. The new function must compute the derivative according to the article from Thelen (2003) (Eq. (6) and (7)):
  
    + Thelen D; Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older adults (2003)
    
### Lecture 11
  
 *  Discussion on  * BEAR, M. F.; W, C. B.; PARADISO, M. A. Neuroscience: Exploring the brain. 4. ed. Philadelphia, PA, USA: Lippincott Williams & Wilkins, 2015.  (Chapter 13)
 
 Task (for Lecture 12):
 
 * Implement the activation dynamics as in Thelen (2003)
 
### Lecture 12
 
 * Activation Dynamics and Pennation angle
 
 Task (for Lecture 13):  
 
 * Implement the Knee simulation of the Nigg and Herzog's book (chapter 4.8.6, [knee.m](http://isbweb.org/~tgcs/resources/software/bogert/knee.m)) in Python using the muscle model implemented in class. You can use your own model or the contained on this [link](https://github.com/BMClab/bmc/blob/af01ef219d8634d22f8d577dd63ffff7b4691487/courses/modsim2018/renatowatanabe/MuscleModel.ipynb).
 
### Lecture 13
 
 * Knee joint model (adapted from Nigg and Herzog's book)  
   + Compare the force and vellocity of the active contractile element for the different moment arms  
 * Ankle joint model  
   + See Rajagopal et al. (2015) for a list of muscle parameters used in a recent OpenSim model
 
 Task (for Lecture 14):
 
* Implement a simulation of the ankle joint model using the parameters from Thelen (2003) and Elias (2014)

### Lecture 14
 
 * Explore the ankle joint model  
 * [Tutorial on classes in Python ](https://panda.ime.usp.br/pensepy/static/pensepy/13-Classes/classesintro.html)  
 * Convert muscle model to class
 
 Task (for Lecture 15):
 
 * Write a Python Class to implement the muscle model developed during the course. [You can use this notebook](https://github.com/BMClab/bmc/blob/master/courses/modsim2018/renatowatanabe/AnkleModel_aula_15.ipynb) and continue the implememtation of the Class.
 
### Lecture 15
 
 * The muscle function into a Python class
  
### Lecture 16
 
 * Control of an inverted pendulum model with muscle actuators (using a PID controller) based on ankle angle

### Lecture 17
  
 * Control of an inverted pendulum model with muscle actuators (using a PID controller) based on muscle length
 
### Lecture 18
  
 * [Optimization](http://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/Optimization.ipynb)
 
 Task (for Lecture 19):
 
 * Solve problemas 1 and 2 of the notebook above.
 
### Lecture 19
 
 * [Optimization](http://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/Optimization.ipynb)
 
 Task (for Lecture 20):
 
 * Solve problemas 3 and 4 of the notebook above.
 
### Lecture 20
 
 * [Multibody Dynamics](http://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/MultibodyDynamics.ipynb)
 
### Lecture 21
 
 * [Multibody Dynamics](http://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/MultibodyDynamics.ipynb)
 
 Task (for Lecture 23):
 
 * Implement the numerical simulationn of the double pendulum with joint actuators (solve numerically the equation of motion using forward dynamics). 
  
### Lecture 22  

 * Motion capture data of running (3D kinematics and kinetics) in the [BMClab](http://demotu.org/). [Get the data here](https://github.com/BMClab/bmc/tree/master/data/opensim).  
 
 Task (for Lecture 24):  
 
 * Run the inverse dynamics and static optimization analyses for the experimental data collected. 
 
### Lecture 23  

 * Modeling and simulation of the data collected with OpenSim  
 
### Lecture 24  

 * Modeling and simulation of the data collected with OpenSim  
