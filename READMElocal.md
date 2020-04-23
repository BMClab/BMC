# BMC - Notes on Scientific Computing for Biomechanics and Motor Control

Marcos Duarte and Renato Watanabe

This repository is a collection of lecture notes and code on scientific computing and data analysis for Biomechanics and Motor Control.  
These notes (notebooks) are written using [Jupyter Notebook](http://jupyter.org/), part of the [Python ecosystem for scientific computing]( http://scipy.org/).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BMClab/BMC/master)

## Introduction

* [Biomechanics](./notebooks/Biomechanics.ipynb)  
* [The Biomechanics and Motor Control Laboratory @ UFABC](./notebooks/BMClab.ipynb)  

## Scientific programming

* [Python for scientific computing](./notebooks/PythonForScientificComputing.ipynb)  
* [Python tutorial](./notebooks/PythonTutorial.ipynb)
* [Version control with Git and GitHub](./notebooks/VersionControlGitGitHub.ipynb)  
* [Code structure for data analysis](./notebooks/CodeStructure.ipynb)  

## Numerical data analysis

* [Scalar and vector](./notebooks/ScalarVector.ipynb)  
* [Basic trigonometry](./notebooks/TrigonometryBasics.ipynb)  
* [Matrix](./notebooks/Matrix.ipynb)  
* [Descriptive statistics](./notebooks/Statistics-Descriptive.ipynb)  
* [Confidence and prediction intervals](./notebooks/ConfidencePredictionIntervals.ipynb)  
  * [Prediction ellipse and prediction ellipsoid](./notebooks/PredictionEllipseEllipsoid.ipynb)  
* [Curve fitting](./notebooks/CurveFitting.ipynb)  
  * [Polynomial fitting](./notebooks/PolynomialFitting.ipynb)  
* [Propagation of uncertainty](./notebooks/Propagation%20of%20uncertainty.ipynb)  
* Frequency analysis  
  * [Basic properties of signals](./notebooks/SignalBasicProperties.ipynb)  
  * [Fourier series](./notebooks/FourierSeries.ipynb)
  * [Fourier transform](./notebooks/FourierTransform.ipynb)
* [Data filtering in signal processing](./notebooks/DataFiltering.ipynb)  
  * [Residual analysis for the optimal cutoff frequency](./notebooks/ResidualAnalysis.ipynb)  
* [Ordinary Differential Equation](./notebooks/OrdinaryDifferentialEquation.ipynb)  
* [Optimization](./notebooks/Optimization.ipynb)  
* Change detection  
  * [Detection of peaks](./notebooks/DetectPeaks.ipynb)  
  * [Detection of onset](./notebooks/DetectOnset.ipynb)  
  * [Detection of changes using the Cumulative Sum (CUSUM)](./notebooks/DetectCUSUM.ipynb)  
  * [Detection of sequential data](./notebooks/detect_seq.ipynb)  
* [Time normalization of data](./notebooks/TimeNormalization.ipynb)  
* [Ensemble average](./notebooks/EnsembleAverage.ipynb)  
* [Open files in C3D format](./notebooks/OpenC3Dfile.ipynb)  

## Mechanics

### Kinematics

* [Frame of reference](./notebooks/ReferenceFrame.ipynb)  
* [Time-varying frame of reference](./notebooks/Time-varying%20frames.ipynb)
* [Polar and cylindrical frame of reference](./notebooks/PolarBasis.ipynb)
* [Kinematics of particle](./notebooks/KinematicsParticle.ipynb)  
  * [Projectile motion](./notebooks/ProjectileMotion.ipynb)  
  * [Spatial and temporal characteristics](./notebooks/SpatialTemporalCharacteristcs.ipynb)  
  * [Minimum jerk hypothesis](./notebooks/MinimumJerkHypothesis.ipynb)  
* [Kinematics of Rigid Body](./notebooks/KinematicsOfRigidBody.ipynb)  
  * [Angular kinematics (2D)](./notebooks/KinematicsAngular2D.ipynb)  
  * [Kinematic chain](./notebooks/KinematicChain.ipynb)  
  * [Rigid-body transformations (2D)](./notebooks/Transformation2D.ipynb)  
  * [Rigid-body transformations (3D)](./notebooks/Transformation3D.ipynb)  
  * [Determining rigid body transformation using the SVD algorithm](./notebooks/SVDalgorithm.ipynb)  

### Kinetics

* [Fundamental concepts](./notebooks/KineticsFundamentalConcepts.ipynb)  
* [Center of Mass and Moment of Inertia](./notebooks/CenterOfMassAndMomentOfInertia.ipynb)  
* [Newton Laws for particles](./notebooks/newtonLawForParticles.ipynb)
* [Newton-Euler Laws](./notebooks/newton_euler_equations.ipynb)
* [Free body diagram](./notebooks/FreeBodyDiagram.ipynb)
  * [Free body diagram for particles](./notebooks/FBDParticles.ipynb)
  * [Free body diagram for rigid bodies](./notebooks/FreeBodyDiagramForRigidBodies.ipynb)
* [3D Rigid body kinetics](./notebooks/Kinetics3dRigidBody.ipynb)
* [Matrix formalism of the Newton-Euler equations](./notebooks/MatrixFormalism.ipynb)  
* [Lagrangian Mechanics](./notebooks/lagrangian_mechanics.ipynb)  

## Modeling and simulation of human movement

* [Body segment parameters](./notebooks/BodySegmentParameters.ipynb)
* [Muscle modeling](./notebooks/MuscleModeling.ipynb)  
* [Muscle simulation](./notebooks/MuscleSimulation.ipynb)  
* [Musculoskeletal modeling and simulation](./notebooks/MusculoskeletaModelingSimulation.ipynb)  
* [Multibody dynamics of simple biomechanical models](./notebooks/MultibodyDynamics.ipynb)  

## Biomechanical tasks Analysis

* [The inverted pendulum model of the human standing posture](./notebooks/IP_Model.ipynb)
* [Measurements in stabilography](./notebooks/Stabilography.ipynb)  
* [Rambling and Trembling decomposition of the COP](./notebooks/IEP.ipynb)  
* [Biomechanical analysis of vertical jumps](./notebooks/VerticalJump.ipynb)  
* [Gait analysis (2D)](./notebooks/GaitAnalysis2D.ipynb)  
* Force plates  
  * [Kistler force plate calculation](./notebooks/KistlerForcePlateCalculation.ipynb)  
  * [Zebris pressure platform](./notebooks/ReadZebrisPressurePlatformASCIIfiles.ipynb)  

## Electromyography

* [Introduction to data analysis in electromyography](./notebooks/Electromyography.ipynb)  

## How to cite this work

Here is a suggestion to cite this GitHub repository:

> Duarte, M., Watanabe, R.N. (2018) Notes on Scientific Computing for Biomechanics and Motor Control. GitHub repository, <https://github.com/BMClab/BMC>.

And a possible BibTeX entry:

```tex
@misc{Duarte2018,  
    author = {Duarte, M. and Watanabe, R.N.},
    title = {Notes on Scientific Computing for Biomechanics and Motor Control},  
    year = {2018},  
    publisher = {GitHub},  
    journal = {GitHub repository},  
    howpublished = {\url{https://github.com/BMClab/BMC}}  
}
```

## License

The non-software content of this project is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/), and the software code is licensed under the [MIT license](https://opensource.org/licenses/mit-license.php).
