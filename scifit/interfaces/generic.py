"""
Module :py:mod:`scifit.interfaces.generic` defines the class :class:`GenericInterface`
on which any other interfaces must inherit from. This class exposes generic abstract methods
all interfaces must implement.
"""

import inspect

import numpy as np
import pandas as pd

from scipy import optimize

import matplotlib.pyplot as plt

from scifit.errors.base import *


class FitSolverInterface:

    """
    Generic Interface (Abstract Base Class) for all object of the package.
    This class must be subclassed by any other interfaces.

    n experimental points
    m variables
    k parameters

    Min wrt b1, ..., bk of MSE = (1/n)*SUM(f(x1, ..., xm, b1, ..., bk) - y)^2

    x(n,m)
    y(n,1)

    """

    def __init__(self, **kwargs):
        """
        Add configurations switch to instance
        """
        self._configuration = kwargs

    def configuration(self, **kwargs):
        return self._configuration | kwargs

    def store(self, xdata, ydata):
        """
        Validate and store experimental data
        """

        xdata = np.array(xdata)
        ydata = np.array(ydata)

        if xdata.ndim != 2:
            raise InputDataError("Variables must be a two dimensional array")

        if xdata.shape[0] != ydata.shape[0]:
            raise InputDataError("Incompatible shapes between x %s and y %s" % (xdata.shape, ydata.shape))

        if not (np.issubdtype(xdata.dtype, np.number) and np.issubdtype(ydata.dtype, np.number)):
            raise InputDataError("Input values must be numeric")

        if np.any(np.isnan(xdata)) or np.any(np.isnan(ydata)):
            raise InputDataError("Input values cannot contain missing data (NaN)")

        self._xdata = xdata
        self._ydata = ydata

    @staticmethod
    def model(xdata, *parameters):
        """
        Model definition to fit with experimental data.
        This method must be overridden by subclassing
        """
        raise MissingModel("Model is not defined")

    @property
    def observation_space_size(self):
        return self._xdata.shape[0]

    @property
    def n(self):
        return self.observation_space_size

    @property
    def variable_space_size(self):
        return self._xdata.shape[1]

    @property
    def m(self):
        return self.variable_space_size

    @property
    def signature(self):
        return inspect.signature(self.model)

    @property
    def parameter_space_size(self):
        return len(self.signature.parameters) - 1

    @property
    def k(self):
        return self.parameter_space_size

    def solve(self, xdata, ydata, **kwargs):
        """
        Solve fitting problem and return structured solution
        """
        solution = optimize.curve_fit(
            self.model,
            xdata, ydata,
            full_output=True, check_finite=True, nan_policy='raise',
            **self.configuration(**kwargs)
        )
        return {
            "parameters": solution[0],
            "covariance": solution[1],
            "info": solution[2],
            "message": solution[3],
            "status": solution[4]
        }

    def fitted(self, error=False):
        is_fitted = hasattr(self, "_solution")
        if not(is_fitted) and error:
            NotFittedError("Model must be fitted prior to this operation")
        return is_fitted

    def solved(self, error=False):
        has_converged = self.fitted(error=error) and self._solution["status"] in {1, 2, 3, 4}
        if not(has_converged) and error:
            NotSolvedError(
                "Fitting procedure has not converged ({status:}): {message:}".format(**self._solution)
            )
        return has_converged

    def predict(self, xdata):
        if self.fitted(error=True):
            return self.model(xdata, *self._solution["parameters"])

    def score(self, xdata, ydata):
        return np.sum(np.power((self.predict(xdata) - ydata), 2)) / ydata.shape[0]

    def fit(self, xdata, ydata, **kwargs):
        """
        Solve fitting problem and store data and results
        """
        self.store(xdata, ydata)
        self._solution = self.solve(self._xdata, self._ydata, **kwargs)
        self._yhat = self.predict(self._xdata)
        self._score = self.score(self._xdata, self._ydata)
        return self._solution

    def variable_domains(self):
        data = pd.DataFrame(self._xdata)
        return data.describe()

    @staticmethod
    def scale(mode="lin", xmin=0., xmax=1., resolution=101):
        """
        Generate 1-dimensional space
        """
        if mode == "lin":
            return np.linspace(xmin, xmax, resolution)
        elif mode == "log":
            return np.logspace(xmin, xmax, resolution, base=10)
        else:
            raise ConfigurationError("Domain mode must be in {lin, log} got '%s' instead" % mode)

    @classmethod
    def scales(cls, domains, mode="lin", xmin=None, xmax=None, resolution=101):
        scales = [
            cls.scale(
                mode=mode,
                xmin=xmin or domains.loc["min", i],
                xmax=xmax or domains.loc["max", i],
                resolution=resolution
            )
            for i in range(domains.shape[1])
        ]
        return scales

    def variable_scales(self, mode="lin", xmin=None, xmax=None, resolution=101):
        """
        Generate Variables Scales
        """
        return self.scales(self.variable_domains(), mode=mode, xmin=xmin, xmax=xmax, resolution=resolution)

    def variable_space(self, mode="lin", xmin=None, xmax=None, resolution=10):
        """
        Generate Variable Space
        """
        return np.meshgrid(*self.variable_scales(mode=mode, xmin=xmin, xmax=xmax, resolution=resolution))

    def parameter_domains(self, xmin=None, xmax=None):
        """
        Generate Parameter Domains
        """
        if hasattr(self, "_solution"):
            pass
        return

    def parameter_scales(self, mode="lin", xmin=None, xmax=None, resolution=101):
        """
        Generate Parameter Scales
        """
        return self.scales(self.parameter_domains(), mode=mode, xmin=xmin, xmax=xmax, resolution=resolution)

    def parameter_space(self, mode="lin", xmin=0., xmax=1., resolution=10):
        """
        Generate Parameter Space
        """
        scales = [
            self.scale(mode=mode, xmin=xmin, xmax=xmax, resolution=resolution)
            for i in range(self.k)
        ]
        return np.meshgrid(scales)

    def plot_fit(self, title="", resolution=200):
        """
        Plot fit for each variable
        """

        scales = self.variable_scales(resolution=resolution)
        for variable_index, scale in enumerate(scales):

            x = self._xdata[:, variable_index]
            xs = scale.reshape(-1, 1)

            fig, axe = plt.subplots()
            axe.plot(
                x, self._ydata,
                linestyle="none", marker=".",
                label=r"Data: $(x_{{{}}},y)$".format(variable_index) + "\nn={}".format(self.n)
            )
            axe.plot(
                xs, self.predict(xs),
                label=r"Fit: $\hat{{y}} = f(\bar{x},\bar{\beta})$" + "\nMSE={:.3e}".format(self._score)
            )
            axe.set_title("Regression Plot: {}".format(title))
            axe.set_xlabel(r"Independent Variable, $x_{{{}}}$".format(variable_index))
            axe.set_ylabel(r"Dependent Variable, $y$")
            axe.legend()
            axe.grid()

            yield axe

    def plot_mse(self, title="", resolution=200):
        """
        Plot MSE for each parameter pairs
        """
        if self.k > 1:
            pass

        else:
            pass

