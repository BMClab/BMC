# Notes on Scientific Computing for Biomechanics and Motor Control

Interactive guides by Marcos Duarte and Renato Naville Watanabe, [Laboratory of Biomechanics and Motor Control, UFABC](https://bmclab.pesquisa.ufabc.edu.br/).

The notebooks in this repository use [marimo](https://marimo.io/). Each guide combines explanations, executable examples, checkpoint questions, worked exercises, and independent problems with expandable final answers. Python introductions cover basic concepts directly in marimo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23606126.svg)](https://doi.org/10.5281/zenodo.23606126)

## Open a guide

Click a guide title to open its **editable GitHub Pages version**. Click **Run all** to compute results, then edit and rerun cells to experiment. Browser edits do not update this repository; download the Python file to keep your own copy.

Every guide also has a **Read saved outputs** link to its read-only version in [notebooks_html/](notebooks_html/), with saved text, tables and plots. Both modes are hosted on [this repository's GitHub Pages site](https://bmclab.github.io/BMC/), independently of molab and marimo.app. Editable notebooks still download their Python runtime, packages and linked data over the Internet.

The compact **molab** and **marimo** badges remain alternative interactive versions. The [Python sources](notebooks/) can also be downloaded and opened locally or uploaded to either service. GitHub Actions automatically discovers the notebooks, rebuilds the site and publishes editable versions directly at [the site root](https://bmclab.github.io/BMC/) on every push to `master`.

For local use, install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the repository, and run these commands from its root:

```bash
uv sync
uv run marimo edit notebooks/reference_frame.py
```

The repository environment uses Python 3.14 or newer. Each notebook also declares its own dependencies for standalone use. There is no need to install or import `pyodide` in a notebook.

Examples use bundled data when the repository is available. Standalone notebooks download the same public assets into a temporary cache when needed. Keep an Internet connection for those assets and external media; file-format and analysis errors remain visible.

To refresh the saved pages after editing a notebook, use [the export/update script](scripts/README.md). Static snapshots keep their saved results; their display assets and external media still require an Internet connection.

## Introduction

- [Introduction to Biomechanics](https://bmclab.github.io/BMC/biomechanics.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/biomechanics.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/biomechanics.py)
- [The Biomechanics and Motor Control Laboratory](https://bmclab.github.io/BMC/bmclab.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/bmclab.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/bmclab.py)

## Scientific programming

- [How to install Python for scientific computing](https://bmclab.github.io/BMC/python_installation.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/python_installation.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/python_installation.py)
- [Python for scientific computing](https://bmclab.github.io/BMC/python_for_scientific_computing.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/python_for_scientific_computing.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/python_for_scientific_computing.py)
- [Tutorial on Python for scientific computing](https://bmclab.github.io/BMC/python_tutorial.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/python_tutorial.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/python_tutorial.py)
- [Version control with Git and GitHub](https://bmclab.github.io/BMC/version_control_git_github.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/version_control_git_github.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/version_control_git_github.py)
- [Code structure for data analysis](https://bmclab.github.io/BMC/code_structure.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/code_structure.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/code_structure.py)

## Numerical data analysis

- [Scalar and vector](https://bmclab.github.io/BMC/scalar_vector.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/scalar_vector.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/scalar_vector.py)
- [Basic trigonometry](https://bmclab.github.io/BMC/trigonometry_basics.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/trigonometry_basics.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/trigonometry_basics.py)
- [Matrix](https://bmclab.github.io/BMC/matrix.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/matrix.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/matrix.py)
- [Descriptive statistics of data](https://bmclab.github.io/BMC/statistics_descriptive.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/statistics_descriptive.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/statistics_descriptive.py)
- [Bayesian Statistics](https://bmclab.github.io/BMC/statistics_bayesian.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/statistics_bayesian.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/statistics_bayesian.py)
- Confidence and prediction intervals

    - [Confidence and prediction intervals](https://bmclab.github.io/BMC/confidence_prediction_intervals.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/confidence_prediction_intervals.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/confidence_prediction_intervals.py)
    - [Prediction ellipse and prediction ellipsoid](https://bmclab.github.io/BMC/prediction_ellipse_ellipsoid.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/prediction_ellipse_ellipsoid.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/prediction_ellipse_ellipsoid.py)

- Curve fitting

    - [Curve fitting](https://bmclab.github.io/BMC/curve_fitting.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/curve_fitting.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/curve_fitting.py)
    - [Polynomial fitting with confidence/prediction intervals](https://bmclab.github.io/BMC/polynomial_fitting.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/polynomial_fitting.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/polynomial_fitting.py)

- [Propagation of uncertainty (error)](https://bmclab.github.io/BMC/propagation_uncertainty.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/propagation_uncertainty.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/propagation_uncertainty.py)
- Frequency analysis

    - [Basic properties of signals](https://bmclab.github.io/BMC/signal_basic_properties.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/signal_basic_properties.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/signal_basic_properties.py)
    - [Fourier series](https://bmclab.github.io/BMC/fourier_series.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/fourier_series.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/fourier_series.py)
    - [Fourier transform](https://bmclab.github.io/BMC/fourier_transform.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/fourier_transform.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/fourier_transform.py)

- Signal processing

    - [Data filtering in signal processing](https://bmclab.github.io/BMC/data_filtering.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/data_filtering.py)
    - [Residual analysis to determine the optimal cutoff frequency](https://bmclab.github.io/BMC/residual_analysis.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/residual_analysis.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/residual_analysis.py)
    - [Replace missing data](https://bmclab.github.io/BMC/rep_missing.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/rep_missing.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/rep_missing.py)
    - [Resample a sampled signal](https://bmclab.github.io/BMC/resample.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/resample.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/resample.py)

- [Introduction to numerical solution of Ordinary Differential Equation](https://bmclab.github.io/BMC/ordinary_differential_equation.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/ordinary_differential_equation.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/ordinary_differential_equation.py)
- [Optimization](https://bmclab.github.io/BMC/optimization.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/optimization.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/optimization.py)
- [Change detection](https://github.com/demotu/detecta)
- [Time normalization of data](https://github.com/demotu/tnorma)
- [Ensemble average](https://bmclab.github.io/BMC/ensemble_average.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/ensemble_average.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/ensemble_average.py)
- [Select data vectors by similarity using a metric score](https://bmclab.github.io/BMC/similarity.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/similarity.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/similarity.py)
- Data files

    - [Text and binary files](https://bmclab.github.io/BMC/text_and_binary_files.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/text_and_binary_files.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/text_and_binary_files.py)
    - [pandas Data](https://bmclab.github.io/BMC/pandas_data.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/pandas_data.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/pandas_data.py)
    - [Open C3D files](https://bmclab.github.io/BMC/open_c3d_file.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/open_c3d_file.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/open_c3d_file.py)
    - [Read C3D data into labelled tables](https://bmclab.github.io/BMC/read_c3d.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/read_c3d.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/read_c3d.py)
    - [Read Delsys data exported by Cortex](https://bmclab.github.io/BMC/read_delsys_cortex_file.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/read_delsys_cortex_file.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/read_delsys_cortex_file.py)
    - [Read and write Cortex ASCII files](https://bmclab.github.io/BMC/read_mac_ascii_files.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/read_mac_ascii_files.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/read_mac_ascii_files.py)
    - [Read Cortex Motion Analysis Corporation .trc file](https://bmclab.github.io/BMC/read_trc.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/read_trc.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/read_trc.py)
    - [Read Cortex Motion Analysis Corporation .trc and .forces files (example)](https://bmclab.github.io/BMC/read_trc_forces_files.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/read_trc_forces_files.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/read_trc_forces_files.py)

- [Direct Linear Transformation (DLT)](https://bmclab.github.io/BMC/dlt.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/dlt.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/dlt.py)

## Mechanics

### Kinematics

- [Frame of reference](https://bmclab.github.io/BMC/reference_frame.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/reference_frame.py)
- [Determination of a coordinate system](https://bmclab.github.io/BMC/coordinate_system.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/coordinate_system.py)
- [Path frame](https://bmclab.github.io/BMC/path_frame.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/path_frame.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/path_frame.py)
- [Polar and Cylindrical Frame of Reference](https://bmclab.github.io/BMC/polar_basis.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/polar_basis.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/polar_basis.py)
- Particle kinematics

    - [Kinematics of a particle](https://bmclab.github.io/BMC/kinematics_particle.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_particle.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/kinematics_particle.py)
    - [Projectile motion](https://bmclab.github.io/BMC/projectile_motion.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/projectile_motion.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/projectile_motion.py)
    - [Spatial and temporal characteristics of a movement pattern](https://bmclab.github.io/BMC/spatial_temporal_characteristics.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/spatial_temporal_characteristics.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/spatial_temporal_characteristics.py)
    - [The minimum jerk hypothesis](https://bmclab.github.io/BMC/minimum_jerk_hypothesis.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/minimum_jerk_hypothesis.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/minimum_jerk_hypothesis.py)

- Rigid-body kinematics

    - [Angular kinematics in a plane (2D)](https://bmclab.github.io/BMC/kinematics_angular_2d.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_angular_2d.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/kinematics_angular_2d.py)
    - [Kinematic chain in a plane (2D)](https://bmclab.github.io/BMC/kinematic_chain.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematic_chain.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/kinematic_chain.py)
    - [Velocity and acceleration of a point of a rigid body](https://bmclab.github.io/BMC/kinematics_of_rigid_body.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_of_rigid_body.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/kinematics_of_rigid_body.py)
    - [Rigid-body transformations in a plane (2D)](https://bmclab.github.io/BMC/transformation_2d.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_2d.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/transformation_2d.py)
    - [Rigid-body transformations in three dimensions](https://bmclab.github.io/BMC/transformation_3d.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/transformation_3d.py)
    - [Determining rigid body transformation using the SVD algorithm](https://bmclab.github.io/BMC/svd_algorithm.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/svd_algorithm.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/svd_algorithm.py)
    - [Angular velocity in 3D movements](https://bmclab.github.io/BMC/angular_velocity_3d.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/angular_velocity_3d.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/angular_velocity_3d.py)
    - [Rotating an ellipsoid about one axis](https://bmclab.github.io/BMC/ellipsoid_rotation_matrix_3d.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/ellipsoid_rotation_matrix_3d.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/ellipsoid_rotation_matrix_3d.py)
    - [Ellipsoids, rotation sequences and a marker-derived basis](https://bmclab.github.io/BMC/ellipsoid_3d_rotation_matrix_1.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/ellipsoid_3d_rotation_matrix_1.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/ellipsoid_3d_rotation_matrix_1.py)

### Kinetics

- [Kinetics: fundamental concepts](https://bmclab.github.io/BMC/kinetics_fundamental_concepts.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinetics_fundamental_concepts.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/kinetics_fundamental_concepts.py)
- [Center of Mass and Moment of Inertia](https://bmclab.github.io/BMC/center_of_mass_and_moment_of_inertia.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/center_of_mass_and_moment_of_inertia.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/center_of_mass_and_moment_of_inertia.py)
- [Motion of a particle - Newtonian approach](https://bmclab.github.io/BMC/newton_law_for_particles.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/newton_law_for_particles.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/newton_law_for_particles.py)
- [Newton-Euler equations for rigid bodies](https://bmclab.github.io/BMC/newton_euler_equations.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/newton_euler_equations.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/newton_euler_equations.py)
- [Free body diagram](https://bmclab.github.io/BMC/free_body_diagram.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/free_body_diagram.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/free_body_diagram.py)

    - [Free-Body Diagram for particles](https://bmclab.github.io/BMC/fbd_particles.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/fbd_particles.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/fbd_particles.py)
    - [Free Body Diagram for Rigid Bodies](https://bmclab.github.io/BMC/free_body_diagram_for_rigid_bodies.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/free_body_diagram_for_rigid_bodies.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/free_body_diagram_for_rigid_bodies.py)

- [Three-dimensional rigid body kinetics](https://bmclab.github.io/BMC/kinetics_3d_rigid_body.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinetics_3d_rigid_body.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/kinetics_3d_rigid_body.py)
- [Matrix Formalism of the  Equations of Movement](https://bmclab.github.io/BMC/matrix_formalism.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/matrix_formalism.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/matrix_formalism.py)
- [Lagrangian mechanics](https://bmclab.github.io/BMC/lagrangian_mechanics.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/lagrangian_mechanics.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/lagrangian_mechanics.py)
- [Lagrangian mechanics in generalized coordinates](https://bmclab.github.io/BMC/lagrangian_mechanics_generalized.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/lagrangian_mechanics_generalized.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/lagrangian_mechanics_generalized.py)
- [Principle of least action](https://bmclab.github.io/BMC/principle_of_least_action.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/principle_of_least_action.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/principle_of_least_action.py)

## Modeling and simulation of human movement

- [Body segment parameters](https://bmclab.github.io/BMC/body_segment_parameters.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/body_segment_parameters.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/body_segment_parameters.py)
- [Basic Mechanical Elements of a Muscle Model](https://bmclab.github.io/BMC/basic_mechanical_elements_muscle_model.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/basic_mechanical_elements_muscle_model.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/basic_mechanical_elements_muscle_model.py)
- [Viscoelastic Models of Biological Tissues](https://bmclab.github.io/BMC/ch2_viscoelastic_models.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/ch2_viscoelastic_models.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/ch2_viscoelastic_models.py)
- [Muscle modeling](https://bmclab.github.io/BMC/muscle_modeling.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/muscle_modeling.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/muscle_modeling.py)
- [Muscle simulation](https://bmclab.github.io/BMC/muscle_simulation.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/muscle_simulation.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/muscle_simulation.py)
- [Musculoskeletal modeling and simulation](https://bmclab.github.io/BMC/musculoskeletal_modeling_simulation.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/musculoskeletal_modeling_simulation.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/musculoskeletal_modeling_simulation.py)
- [Multibody dynamics of one and two-link systems](https://bmclab.github.io/BMC/multibody_dynamics.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/multibody_dynamics.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/multibody_dynamics.py)
- [Matrix Formalism of the Newton-Euler equations](https://bmclab.github.io/BMC/multibody_inverse_dynamics.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/multibody_inverse_dynamics.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/multibody_inverse_dynamics.py)

## Biomechanical tasks analysis

- [The inverted pendulum model of the human standing](https://bmclab.github.io/BMC/ip_model.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/ip_model.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/ip_model.py)
- [Measurements in stabilography](https://bmclab.github.io/BMC/stabilography.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/stabilography.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/stabilography.py)
- [Postural sway: rambling and trembling](https://bmclab.github.io/BMC/iep.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/iep.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/iep.py)
- [Biomechanical analysis of vertical jumps](https://bmclab.github.io/BMC/vertical_jump.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/vertical_jump.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/vertical_jump.py)
- [Inverse dynamics (2D) for gait analysis](https://bmclab.github.io/BMC/gait_analysis_2d.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/gait_analysis_2d.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/gait_analysis_2d.py)
- Force plates

    - [Algorithm for force plate calibration](https://bmclab.github.io/BMC/force_plate_calibration.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/force_plate_calibration.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/force_plate_calibration.py)
    - [Open AMTI .bsf file](https://bmclab.github.io/BMC/amti_bsf_file.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/amti_bsf_file.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/amti_bsf_file.py)
    - [Kistler force plate calculation](https://bmclab.github.io/BMC/kistler_force_plate_calculation.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kistler_force_plate_calculation.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/kistler_force_plate_calculation.py)
    - [Read Zebris pressure platform ASCII files](https://bmclab.github.io/BMC/read_zebris_pressure_platform_ascii_files.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/read_zebris_pressure_platform_ascii_files.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/read_zebris_pressure_platform_ascii_files.py)

## Electromyography

- [Introduction to data analysis in electromyography](https://bmclab.github.io/BMC/electromyography.html) [![molab][molab-badge]](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/electromyography.py) [![marimo][marimo-badge]](https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/electromyography.py)

## Verification and maintenance

```bash
uv run pytest -q
uv run marimo check --strict notebooks/*.py
```

To execute one notebook headlessly, use `uv run marimo export html notebooks/reference_frame.py -o /tmp/ReferenceFrame.html`. Running a notebook as `python notebook.py` opens its application; it does not verify all cell bodies.

The common repository-file resolver is maintained in `functions/notebook_resources.py` and embedded in standalone notebooks. After changing it, run `uv run python scripts/sync_notebook_resources.py`. Generated session state and saved outputs under `__marimo__/` are ignored and are not shipped with the sources; run the notebooks to generate their outputs.

## How to cite this work

Release metadata for Zenodo is maintained in [.zenodo.json](.zenodo.json). The citation below refers to the archived `v0.1.0` release.

Here is a suggestion to cite this GitHub repository:

> Marcos Duarte and Renato Naville Watanabe. (2026). Notes on Scientific Computing for Biomechanics and Motor Control (Version v0.1.0). Zenodo. <https://doi.org/10.5281/zenodo.23606126>

And a BibTeX entry:

```tex
@misc{marcos_duarte_2026_23606126,
  author       = {Marcos Duarte and Renato Naville Watanabe},
  title        = {{Notes on Scientific Computing for Biomechanics and Motor Control}},
  month        = oct,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23606126},
  url          = {https://doi.org/10.5281/zenodo.23606126}
}
```

## License

The non-software content of this project is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/), and the software code is licensed under the [MIT license](https://opensource.org/licenses/mit-license.php).

[molab-badge]: https://img.shields.io/badge/molab-6757D4?style=flat-square
[marimo-badge]: https://img.shields.io/badge/marimo-2D8A4E?style=flat-square
