Quick Start Guide
=================

Fitting a model to experimental data with scifit.

Classic Linear Regression
-------------------------

First import the Solver interface:

    .. code-block:: python

        from scifit.interfaces.generic import FitSolverInterface

Then create the model function by overridding the model method:

    .. code-block:: python

        class LinearFitSolver(FitSolverInterface):
            @staticmethod
            def model(x, a, b):
                return a * x[:, 0] + b

Now create an new instance of the solver:

    .. code-block:: python

        solver = LinearFitSolver()

And fit to experimental data:

    .. code-block:: python

        solution = solver.fit(X, y, sigma=sigma)

It returns a complete solution set.
And then exposes convenience to analyse the regression in depth:

    .. code-block:: python

        solver.plot_fit()
        solver.plot_loss()

Rendering respectively the fitted model to the data and the loss function
wrt parameters:

.. image:: ../media/figures/QuickStart_LinearFit.png
  :width: 560
  :alt: Classical Linear Regression (fit)

.. image:: ../media/figures/QuickStart_LinearLoss.png
  :width: 560
  :alt: Classical Linear Regression (loss)
