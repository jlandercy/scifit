# Changes

## To-do list

 - [ ] Implement ODE Solver (special case for Kinetics) make it FitSolvable as well
 - [ ] Implement complex models (see StackOverflow 10 years feed of fit and personal website)
 - [ ] Implement specials

## v0.1.3

 - [x] Added special solvers: Debye

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
