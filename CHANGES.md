# Changes

## To-do list

 - [ ] Implement ODE Solver (special case for Kinetics) make it FitSolvable as well
 - [ ] Implement complex models (see StackOverflow 10 years feed of fit and personal website)
 - [ ] Build GitLab CI/CD Pipeline
 - [ ] Build GitHub CI/CD Pipeline
 
## v0.1

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
