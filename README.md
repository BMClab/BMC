# Notes on Scientific Computing for Biomechanics and Motor Control

Interactive guides by Marcos Duarte and Renato Naville Watanabe, [Laboratory of Biomechanics and Motor Control, UFABC](https://bmclab.pesquisa.ufabc.edu.br/).

The notebooks in this repository use [marimo](https://marimo.io/). Each guide combines explanations, executable examples, checkpoint questions, worked exercises, and independent problems with expandable final answers. Python introductions cover basic concepts directly in marimo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23606126.svg)](https://doi.org/10.5281/zenodo.23606126)

## Open a guide

Click a guide title to open its **static HTML page**, with saved text, tables and plots. You can also [browse all guides](https://bmclab.github.io/BMC/notebooks_html/).

Each page's header offers five choices:

- **Static version**: read the saved outputs.
- **Download .py**: save the corresponding marimo Python notebook for local use.
- **Edit in browser**: edit and run the notebook in your current browser.
- **molab**: open and run the notebook in molab.
- **marimo.app**: open and run the notebook in marimo.app.

In the browser editor, click **Run all** to compute results, then edit and rerun cells to experiment. It downloads the Python runtime, packages and linked data over the Internet. Browser edits do not update this repository; download the Python file to keep your own copy.

The [Python sources](notebooks/) can also be downloaded and opened locally. GitHub Actions automatically discovers the notebooks, rebuilds the site and publishes saved and editable versions on every push to `master`.

For local use, install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the repository, and run these commands from its root:

```bash
uv sync
uv run marimo edit notebooks/reference_frame.py
```

The repository environment uses Python 3.14 or newer. Each notebook also declares its own dependencies for standalone use. There is no need to install or import `pyodide` in a notebook.

Examples use bundled data when the repository is available. Standalone notebooks download the same public assets into a temporary cache when needed. Keep an Internet connection for those assets and external media; file-format and analysis errors remain visible.

To refresh the saved pages after editing a notebook, use [the export/update script](scripts/README.md). Static snapshots keep their saved results; their display assets and external media still require an Internet connection.

## Introduction

- [Introduction to Biomechanics](https://bmclab.github.io/BMC/notebooks_html/biomechanics.html)
- [The Biomechanics and Motor Control Laboratory](https://bmclab.github.io/BMC/notebooks_html/bmclab.html)

## Scientific programming

- [How to install Python for scientific computing](https://bmclab.github.io/BMC/notebooks_html/python_installation.html)
- [Python for scientific computing](https://bmclab.github.io/BMC/notebooks_html/python_for_scientific_computing.html)
- [Tutorial on Python for scientific computing](https://bmclab.github.io/BMC/notebooks_html/python_tutorial.html)
- [Version control with Git and GitHub](https://bmclab.github.io/BMC/notebooks_html/version_control_git_github.html)
- [Code structure for data analysis](https://bmclab.github.io/BMC/notebooks_html/code_structure.html)

## Numerical data analysis

- [Scalar and vector](https://bmclab.github.io/BMC/notebooks_html/scalar_vector.html)
- [Basic trigonometry](https://bmclab.github.io/BMC/notebooks_html/trigonometry_basics.html)
- [Matrix](https://bmclab.github.io/BMC/notebooks_html/matrix.html)
- [Descriptive statistics of data](https://bmclab.github.io/BMC/notebooks_html/statistics_descriptive.html)
- [Bayesian Statistics](https://bmclab.github.io/BMC/notebooks_html/statistics_bayesian.html)
- Confidence and prediction intervals
    - [Confidence and prediction intervals](https://bmclab.github.io/BMC/notebooks_html/confidence_prediction_intervals.html)
    - [Prediction ellipse and prediction ellipsoid](https://bmclab.github.io/BMC/notebooks_html/prediction_ellipse_ellipsoid.html)
- Curve fitting
    - [Curve fitting](https://bmclab.github.io/BMC/notebooks_html/curve_fitting.html)
    - [Polynomial fitting with confidence/prediction intervals](https://bmclab.github.io/BMC/notebooks_html/polynomial_fitting.html)
- [Propagation of uncertainty (error)](https://bmclab.github.io/BMC/notebooks_html/propagation_uncertainty.html)
- Frequency analysis
    - [Basic properties of signals](https://bmclab.github.io/BMC/notebooks_html/signal_basic_properties.html)
    - [Fourier series](https://bmclab.github.io/BMC/notebooks_html/fourier_series.html)
    - [Fourier transform](https://bmclab.github.io/BMC/notebooks_html/fourier_transform.html)
- Signal processing
    - [Data filtering in signal processing](https://bmclab.github.io/BMC/notebooks_html/data_filtering.html)
    - [Residual analysis to determine the optimal cutoff frequency](https://bmclab.github.io/BMC/notebooks_html/residual_analysis.html)
    - [Replace missing data](https://bmclab.github.io/BMC/notebooks_html/rep_missing.html)
    - [Resample a sampled signal](https://bmclab.github.io/BMC/notebooks_html/resample.html)
- [Introduction to numerical solution of Ordinary Differential Equation](https://bmclab.github.io/BMC/notebooks_html/ordinary_differential_equation.html)
- [Optimization](https://bmclab.github.io/BMC/notebooks_html/optimization.html)
- [Change detection](https://github.com/demotu/detecta)
- [Time normalization of data](https://github.com/demotu/tnorma)
- [Ensemble average](https://bmclab.github.io/BMC/notebooks_html/ensemble_average.html)
- [Select data vectors by similarity using a metric score](https://bmclab.github.io/BMC/notebooks_html/similarity.html)
- Data files
    - [Text and binary files](https://bmclab.github.io/BMC/notebooks_html/text_and_binary_files.html)
    - [pandas Data](https://bmclab.github.io/BMC/notebooks_html/pandas_data.html)
    - [Open C3D files](https://bmclab.github.io/BMC/notebooks_html/open_c3d_file.html)
    - [Read C3D data into labelled tables](https://bmclab.github.io/BMC/notebooks_html/read_c3d.html)
    - [Read Delsys data exported by Cortex](https://bmclab.github.io/BMC/notebooks_html/read_delsys_cortex_file.html)
    - [Read and write Cortex ASCII files](https://bmclab.github.io/BMC/notebooks_html/read_mac_ascii_files.html)
    - [Read Cortex Motion Analysis Corporation .trc file](https://bmclab.github.io/BMC/notebooks_html/read_trc.html)
    - [Read Cortex Motion Analysis Corporation .trc and .forces files (example)](https://bmclab.github.io/BMC/notebooks_html/read_trc_forces_files.html)
- [Direct Linear Transformation (DLT)](https://bmclab.github.io/BMC/notebooks_html/dlt.html)

## Mechanics

### Kinematics

- [Frame of reference](https://bmclab.github.io/BMC/notebooks_html/reference_frame.html)
- [Determination of a coordinate system](https://bmclab.github.io/BMC/notebooks_html/coordinate_system.html)
- [Path frame](https://bmclab.github.io/BMC/notebooks_html/path_frame.html)
- [Polar and Cylindrical Frame of Reference](https://bmclab.github.io/BMC/notebooks_html/polar_basis.html)
- Particle kinematics
    - [Kinematics of a particle](https://bmclab.github.io/BMC/notebooks_html/kinematics_particle.html)
    - [Projectile motion](https://bmclab.github.io/BMC/notebooks_html/projectile_motion.html)
    - [Spatial and temporal characteristics of a movement pattern](https://bmclab.github.io/BMC/notebooks_html/spatial_temporal_characteristics.html)
    - [The minimum jerk hypothesis](https://bmclab.github.io/BMC/notebooks_html/minimum_jerk_hypothesis.html)
- Rigid-body kinematics
    - [Angular kinematics in a plane (2D)](https://bmclab.github.io/BMC/notebooks_html/kinematics_angular_2d.html)
    - [Kinematic chain in a plane (2D)](https://bmclab.github.io/BMC/notebooks_html/kinematic_chain.html)
    - [Velocity and acceleration of a point of a rigid body](https://bmclab.github.io/BMC/notebooks_html/kinematics_of_rigid_body.html)
    - [Rigid-body transformations in a plane (2D)](https://bmclab.github.io/BMC/notebooks_html/transformation_2d.html)
    - [Rigid-body transformations in three dimensions](https://bmclab.github.io/BMC/notebooks_html/transformation_3d.html)
    - [Determining rigid body transformation using the SVD algorithm](https://bmclab.github.io/BMC/notebooks_html/svd_algorithm.html)
    - [Angular velocity in 3D movements](https://bmclab.github.io/BMC/notebooks_html/angular_velocity_3d.html)
    - [Rotating an ellipsoid about one axis](https://bmclab.github.io/BMC/notebooks_html/ellipsoid_rotation_matrix_3d.html)
    - [Ellipsoids, rotation sequences and a marker-derived basis](https://bmclab.github.io/BMC/notebooks_html/ellipsoid_3d_rotation_matrix_1.html)

### Kinetics

- [Kinetics: fundamental concepts](https://bmclab.github.io/BMC/notebooks_html/kinetics_fundamental_concepts.html)
- [Center of Mass and Moment of Inertia](https://bmclab.github.io/BMC/notebooks_html/center_of_mass_and_moment_of_inertia.html)
- [Motion of a particle - Newtonian approach](https://bmclab.github.io/BMC/notebooks_html/newton_law_for_particles.html)
- [Newton-Euler equations for rigid bodies](https://bmclab.github.io/BMC/notebooks_html/newton_euler_equations.html)
- [Free body diagram](https://bmclab.github.io/BMC/notebooks_html/free_body_diagram.html)
    - [Free-Body Diagram for particles](https://bmclab.github.io/BMC/notebooks_html/fbd_particles.html)
    - [Free Body Diagram for Rigid Bodies](https://bmclab.github.io/BMC/notebooks_html/free_body_diagram_for_rigid_bodies.html)
- [Three-dimensional rigid body kinetics](https://bmclab.github.io/BMC/notebooks_html/kinetics_3d_rigid_body.html)
- [Matrix Formalism of the  Equations of Movement](https://bmclab.github.io/BMC/notebooks_html/matrix_formalism.html)
- [Lagrangian mechanics](https://bmclab.github.io/BMC/notebooks_html/lagrangian_mechanics.html)
- [Lagrangian mechanics in generalized coordinates](https://bmclab.github.io/BMC/notebooks_html/lagrangian_mechanics_generalized.html)
- [Principle of least action](https://bmclab.github.io/BMC/notebooks_html/principle_of_least_action.html)

## Modeling and simulation of human movement

- [Body segment parameters](https://bmclab.github.io/BMC/notebooks_html/body_segment_parameters.html)
- [Basic Mechanical Elements of a Muscle Model](https://bmclab.github.io/BMC/notebooks_html/basic_mechanical_elements_muscle_model.html)
- [Viscoelastic Models of Biological Tissues](https://bmclab.github.io/BMC/notebooks_html/ch2_viscoelastic_models.html)
- [Muscle modeling](https://bmclab.github.io/BMC/notebooks_html/muscle_modeling.html)
- [Muscle simulation](https://bmclab.github.io/BMC/notebooks_html/muscle_simulation.html)
- [Musculoskeletal modeling and simulation](https://bmclab.github.io/BMC/notebooks_html/musculoskeletal_modeling_simulation.html)
- [Multibody dynamics of one and two-link systems](https://bmclab.github.io/BMC/notebooks_html/multibody_dynamics.html)
- [Matrix Formalism of the Newton-Euler equations](https://bmclab.github.io/BMC/notebooks_html/multibody_inverse_dynamics.html)

## Biomechanical tasks analysis

- [The inverted pendulum model of the human standing](https://bmclab.github.io/BMC/notebooks_html/ip_model.html)
- [Measurements in stabilography](https://bmclab.github.io/BMC/notebooks_html/stabilography.html)
- [Postural sway: rambling and trembling](https://bmclab.github.io/BMC/notebooks_html/iep.html)
- [Biomechanical analysis of vertical jumps](https://bmclab.github.io/BMC/notebooks_html/vertical_jump.html)
- [Inverse dynamics (2D) for gait analysis](https://bmclab.github.io/BMC/notebooks_html/gait_analysis_2d.html)
- Force plates
    - [Algorithm for force plate calibration](https://bmclab.github.io/BMC/notebooks_html/force_plate_calibration.html)
    - [Open AMTI .bsf file](https://bmclab.github.io/BMC/notebooks_html/amti_bsf_file.html)
    - [Kistler force plate calculation](https://bmclab.github.io/BMC/notebooks_html/kistler_force_plate_calculation.html)
    - [Read Zebris pressure platform ASCII files](https://bmclab.github.io/BMC/notebooks_html/read_zebris_pressure_platform_ascii_files.html)

## Electromyography

- [Introduction to data analysis in electromyography](https://bmclab.github.io/BMC/notebooks_html/electromyography.html)

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
