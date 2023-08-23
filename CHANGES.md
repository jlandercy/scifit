# Changes

## To-do list

 - [ ] Implement ODE Solver (special case for Kinetics) make it FitSolvable as well
 - [ ] Implement complex models (see StackOverflow 10 years feed of fit and personal website)
   - [ ] Kinetics
   - [ ] Mechanics
   - [ ] Gradient recovery
   - [ ] Distribution fit
 - [ ] Print parameters in scientific format if too small or too large
 - [ ] Take time to refactor load and store to make it compliant with workflow

SO ``curve-fitting`` tag:

 - https://stackoverflow.com/questions/76603587/non-linear-data-fitting-for-kinetic-data/76702800#76702800
 - https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
 - https://stackoverflow.com/questions/73814378/scipy-curve-fit-incorrect-for-large-x-values/73817184#73817184
 - https://stackoverflow.com/questions/73365651/instability-in-fitting-data-using-scipy-optimize-library/73369630#73369630
 - https://stackoverflow.com/questions/68523795/fit-a-custom-function-in-python/68526879#68526879
 - https://stackoverflow.com/questions/70278957/python-fitting-curve-with-integral-func/70694744#70694744
 - https://stackoverflow.com/questions/73891034/how-to-estimate-confidence-intervals-beyond-the-current-simulated-step-based-on/73891943#73891943
 - https://stackoverflow.com/questions/63637144/python-rayleigh-fit-histogram/63646040#63646040



## v0.1.7

 - [x] Added Docker GitHub Action workflow to compile documentation

## v0.1.6

 - [x] Changed store interface to make it more consistent with load
 - [x] Adapted how automatic parameter domains are computed
 - [x] Adapted parameters domains to make loss figures more wide and interpretable
 - [x] Starting parameters iterations pathways on loss figures
 - [x] Added parameters domain from gradient descent iterations

## v0.1.5

 - [x] Added serialization tests to ensure solution continuity among dumps
 - [x] Added seed reproducibility for dataset generation
 
## v0.1.4

 - [x] Added real data based test as well as pure synthetic
 - [x] Added load and dump function to exchange standardized CSV
 - [x] Corrected sigma management when only scalar or None are used inplace of array
 - [x] Added synthetic dataset generation, bound for unit test operations
 - [x] Started test with real dataset 

## v0.1.3

 - [x] Added special solvers: Debye Heat Capacity, Crank Diffusion, Raney Keton Dehydrogenation
 - [x] Added log scale mode for fit and loss figures
 - [x] Updated nox commands for building and cache
 - [x] Updated notebooks and documentation

## v0.1.2

 - [x] Added error surface in addition with contour levels
 - [x] Added book test suite for resources generation
 - [x] Added dataset export from solver interface
 - [x] Updated CI pipeline
 - [x] Corrected typo in figures
 - [x] Adapted quality test to be more realistic
 - [x] Added fake sigma capability to tests bad chi square regression
 - [x] Completed publication workflow on PyPi

## v0.1.1

 - [x] Added lot of logistic solvers
 - [x] Added minimizer to check `curve_fit`
 - [x] Added CI pipeline for GitHub

## v0.1.0

 - [x] First beta version of the package
 - [x] Created first solver interface for fitting problems (`FitSolverInterface`)
   - [x] Complete fitting procedure with `SciKit-Learn` compliant interface
   - [x] Implemented Chi Square Goodness of Fit tests for fitting procedure
 - [x] Created a bunch of fit model (linear and scientific namespaces)
 - [x] Created a bunch of tests to assess capabilities
 - [x] Created summary figures:
   - [x] Fit Plot to check adjustment
   - [x] Loss Plot to check parameters convergence and uniqueness (low dimensional and scatter)
   - [x] Chi Square Goodness of Fit Plot to check fitting compliance
 - [x] Started Sphinx documentation for whole package
 - [x] Created quick start guide using Jupyter notebooks
