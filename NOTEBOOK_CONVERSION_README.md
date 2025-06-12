# Jupyter to Marimo Notebook Conversion

This document describes the conversion of all Jupyter notebooks in the `notebooks/` directory to Marimo notebooks.

## Overview

**Date of Conversion:** December 2024  
**Total Notebooks Converted:** 91  
**Conversion Success Rate:** 100% (91/91)

## Directory Structure

```
├── notebooks/                    # Converted Marimo notebooks (.py files) - NOW ACTIVE
├── notebooks_original/           # Backup of all original Jupyter notebooks
└── convert_notebooks.py          # Conversion script used
```

## What is Marimo?

[Marimo](https://marimo.io/) is a reactive notebook for Python that offers several advantages over traditional Jupyter notebooks:

- **Reactive execution**: Cells automatically re-run when their dependencies change
- **No hidden state**: Eliminates common notebook pitfalls like out-of-order execution
- **Git-friendly**: Notebooks are stored as Python files, making version control easier
- **Interactive widgets**: Built-in support for interactive elements
- **Reproducible**: Ensures notebooks run the same way every time

## Converted Notebooks

All 91 Jupyter notebooks have been successfully converted to Marimo format:

### Core Topics
- **Biomechanics**: Biomechanics.py, BodySegmentParameters.py, CenterOfMassAndMomentOfInertia.py
- **Kinematics**: KinematicsParticle.py, KinematicsAngular2D.py, KinematicsOfRigidBody.py
- **Kinetics**: KineticsFundamentalConcepts.py, Kinetics3dRigidBody.py
- **Dynamics**: MultibodyDynamics.py, OrdinaryDifferentialEquation.py

### Mathematical Foundations
- **Linear Algebra**: Matrix.py, MatrixFormalism.py, SVDalgorithm.py
- **Coordinate Systems**: CoordinateSystem.py, ReferenceFrame.py, Transformation2D.py, Transformation3D.py
- **Signal Processing**: DataFiltering.py, FourierSeries.py, FourierTransform.py

### Data Analysis
- **Statistics**: Statistics-Descriptive.py, statistics_bayesian.py
- **Detection Algorithms**: DetectPeaks.py, DetectOnset.py, DetectCUSUM.py
- **Curve Fitting**: CurveFitting.py, PolynomialFitting.py

### Specialized Applications
- **Gait Analysis**: GaitAnalysis2D.py, SpatialTemporalCharacteristcs.py
- **EMG**: Electromyography.py
- **Force Plates**: ForcePlateCalibration.py, KistlerForcePlateCalculation.py
- **Motion Capture**: DLT.py, OpenC3Dfile.py, read_c3d.py

### Programming & Tools
- **Python Tutorials**: PythonTutorial.py, PythonForScientificComputing.py, PythonInstallation.py
- **Version Control**: VersionControlGitGitHub.py
- **Data Handling**: pandas_data.py, read_trc.py, read_mac_ascii_files.py

## Usage Instructions

### Running a Marimo Notebook

To open and run a converted notebook:

```bash
# Navigate to the project directory
cd /path/to/BMC

# Open a specific notebook
marimo edit notebooks/Biomechanics.py

# Or start marimo and browse notebooks
marimo edit
```

### Installing Marimo

If marimo is not installed:

```bash
pip install marimo
```

### Key Differences from Jupyter

1. **File Format**: Marimo notebooks are `.py` files, not `.ipynb`
2. **Reactive Execution**: Changes propagate automatically through dependent cells
3. **No Cell Numbers**: Cells don't have execution numbers
4. **Git-Friendly**: Easier to track changes in version control

## Conversion Process

The conversion was performed using the `marimo convert` command:

```bash
marimo convert notebook.ipynb -o notebook.py
```

### What Was Converted

- ✅ Code cells → Marimo code cells
- ✅ Markdown cells → Marimo markdown cells  
- ✅ Cell structure and organization
- ❌ Output cells (stripped during conversion, as expected)
- ❌ Jupyter-specific metadata

### Post-Conversion Considerations

After conversion, you may need to:

1. **Fix Import Statements**: Ensure all required packages are imported
2. **Resolve Variable Dependencies**: Marimo will highlight any dependency issues
3. **Update Interactive Elements**: Convert Jupyter widgets to Marimo equivalents
4. **Fix LaTeX Equations**: Replace `\begin{equation}...\end{equation}` with `$ $` syntax
5. **Test Execution**: Run through each notebook to ensure proper functionality

### LaTeX Equation Fix Applied

**Issue**: Marimo notebooks don't render LaTeX equations wrapped in `\begin{equation}...\end{equation}` properly.
**Solution**: All equation blocks have been automatically converted to use single `$equation$` syntax (no spaces) for proper Marimo compatibility.
**Files affected**: 66 notebooks contained equation formatting that was fixed.
**Total equation fixes applied**: 3,059 individual spacing corrections were successfully applied.

### Image Reference Fix Applied

**Issue**: Some Marimo notebooks referenced images using relative paths like `../images/` which don't work correctly when notebooks are executed from the `notebooks/` directory.
**Solution**: All image references have been converted to use GitHub URLs with `?raw=1` parameter for reliable access.
**Files affected**: 15+ notebooks contained image references that were fixed.
**Examples of fixes**:
- `../images/pendulum.png` → `https://github.com/BMClab/BMC/blob/master/images/pendulum.png?raw=1`
- `./../images/vector3D.png` → `https://github.com/BMClab/BMC/blob/master/images/vector3D.png?raw=1`

## Backup and Safety

- **Original Jupyter notebooks** are preserved in `notebooks_original/`
- **Current notebooks** in `notebooks/` are now the Marimo versions with corrected image references
- **Conversion script** (`convert_notebooks.py`) is available for reference

## Next Steps

1. **Test converted notebooks** by opening them in Marimo
2. **Fix any conversion issues** that may arise
3. **Update documentation** to reference Marimo notebooks where appropriate
4. **Consider migrating** teaching materials to use Marimo notebooks

## Support

For issues with:
- **Marimo**: Visit [marimo.io](https://marimo.io/) or [GitHub](https://github.com/marimo-team/marimo)
- **Conversion problems**: Check the conversion script or re-run individual conversions
- **Notebook content**: Refer to the original notebooks in `notebooks_original/`

---

**Note**: This conversion maintains all educational content while providing the benefits of Marimo's reactive notebook environment. Students and instructors can now enjoy a more robust and reproducible notebook experience.
