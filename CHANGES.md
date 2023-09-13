# Changes

## To-do list

 - [x] Implement ODE Solver (special case for Kinetics) make it FitSolvable as well
 - [ ] Implement complex models (see StackOverflow 10 years feed of fit and personal website)
   - [x] Kinetics
   - [ ] Mechanics
   - [ ] Gradient recovery
   - [ ] Distribution fit
 - [ ] Refactor all models to get parameters name:
   - [ ] Starting by 0 and get compliant with natural index
   - [x] Write all _model_equation
 - [ ] Apply 10^log10 to model for Kinetic in order to reduce errors
 - [ ] Find out why some conversion ratio are not shown in steady mode
 - [ ] Solve issue when KineticFit Solver p0 is too far away from real give shape error
   - Due to ODE Solver interruption (not converging stop before reaching full t_eval)
 - [ ] Implement SIR model and fit epidemic data (introduce bootstrap)
 - [ ] Add transformers (Log-Log, Lineweaver-Burk, Standardizer)
 - [ ] Add Pipeline object
 - [ ] Chromatogram Solver:
   - [ ] Solve why peak are still grouped in some conditions
   - [ ] Create 10 datasets as use cases
   - [ ] Create solver with different mode/settings

## v0.1.15

 - [x] Kinetic rates are now computed from model instead of deriving solutions 
 - [x] Refactored namespace to be more sensical
 - [x] Added Chromatogram Solver to detect peaks
 - [x] Added Baseline Noise, LOD and LOQ estimation

## v0.1.14

 - [x] Validated ODE matrix system against manually writen ODE systems
 - [x] Added Brusselator Kinetic and Fit Solver
 - [x] Added steady state capability to Kinetic Solver
 - [x] Added Quotient Rate figure
 - [x] Corrected mislabeling when selecting substances (eg.: A, B instead of E, F)
 - [x] Refactoring of solver namespace
 - [x] Stabilized precision for derivatives
 - [x] Creating report for Kinetic Solver
 
## v0.1.13

 - [x] Added Activated State Model Kinetic solver:
   - [x] Capable of direct, indirect and equilibrium reactions
   - [x] Capable of multiple kinetics
   - [x] Capable of auto-catalytic kinetics
   - [x] Added Reaction Quotient
   - [x] Added Conversion Ratio
   - [x] Added Concentration velocities and accelerations
   - [x] Added Instantaneous Selectivity
   - [x] Succeeded to bind KineticSolver w/ FitSolver
   - [x] Add reference index for substance
   - [x] Levenspiel diagram

## v0.1.12

 - [x] Remove dimension argument from scales
 - [x] Started automatic report with MD/LaTeX/PDF
   - [x] With Jinja templating
   - [x] Binary stream ready for figure and table
   - [x] Inline SVG through MD figure
   - [x] Added longtable and siunitx into the game
 - [x] Added MD/docx report export
 - [x] Added MD/HTML report export

## v0.1.11

 - [x] Created mixins to generalize interfaces
 - [x] Migrated part of generic FitSolver interface to mixins
 - [x] Breaking: Changed FitSolver interface data life cycle to make it compliant with sklearn life cycle
 - [x] Updated load, fit, refit definition and behaviours

## v0.1.10

 - [x] Updated documentation

## v0.1.9

 - [x] Regex to adapt requirements and remove windows dev traces
 - [x] Added smart display for parameters
 - [x] Added Gaussian Peak model
 - [x] Added Exponential Modified Gaussian Peak model
 - [x] Added Linear Squared Slope model (ill-defined)
 - [x] Add a Kolmogorov Smirnov Tests in addition of Chi Square Test
 - [x] Surfaces are available in plot_loss as well

## v0.1.8
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
