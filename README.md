[![Pypi Workflow](https://github.com/jlandercy/scifit/actions/workflows/pypi.yaml/badge.svg?branch=main)](https://github.com/jlandercy/scifit/actions/workflows/pypi.yaml)
[![Documentations Workflow](https://github.com/jlandercy/scifit/actions/workflows/docs.yaml/badge.svg?branch=main)](https://github.com/jlandercy/scifit/actions/workflows/docs.yaml)

![SciFit Banner](./docs/source/media/branding/Banner.png)

# SciFit

> Comprehensive fits for scientists

Welcome to SciFit project the Python package for comprehensive fits for scientists
designed to ease fitting procedure and automatically perform the quality assessment.

The SciFit project aims to support your work by:

 - Providing a clean, stable and compliant interface for each solver;
 - Perform ad hoc transformations, processing and tests on each stage of a solver procedure;
 - Render high quality figures summarizing solver solution and the quality assessment.

## Installation

You can install the SciFit package by issuing:

```commandline
python -m pip install --upgrade scifit
```

Which update you to the latest version of the package.

## Quick start

Let's fit some data:

```python
from scifit.solvers.scientific import *

# Select a specific solver:
solver = HillEquationFitSolver()

# Create some synthetic dataset:
data = solver.synthetic_dataset(
    xmin=0.0, xmax=5.0, resolution=50,
    parameters=[3.75, 0.21],
    sigma=0.1, scale_mode="auto", seed=1234,
)

# Perform regression:
solution = solver.fit(data)

# Render results:
axe = solver.plot_fit()
```

We have a nice adjustments for such noisy data:

![Fit figure](./docs/source/media/branding/FitFigure.png)

Or even better a full fit report to check each key points at once:

```python
solver.report("hill_report")
```

![Report page](./docs/source/media/branding/ReportPage.png)

Which produces a PDF file called `hill_report.pdf`.

## Resources

 - [Documentations][20]
 - [Repository][21]

[20]: https://github.com/jlandercy/scifit/tree/main/docs
[21]: https://github.com/jlandercy/scifit