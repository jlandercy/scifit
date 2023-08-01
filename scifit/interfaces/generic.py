"""
Module :py:mod:`scifit.interfaces.generic` defines the class :class:`GenericInterface`
on which any other interfaces must inherit from. This class exposes generic abstract methods
all interfaces must implement.
"""

import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt

from scifit.errors.base import *


class FitSolverInterface:

    """
    Generic Interface (Abstract Base Class) for all object of the package.
    This class must be subclassed by any other interfaces.
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
        Store experimental data for convenience
        """

        xdata = np.array(xdata)
        ydata = np.array(ydata)

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
        raise MissingModel("Model not defined")

    def solve(self, xdata, ydata, **kwargs):
        """
        Solve fitting problem and return structured solution
        """
        solution = optimize.curve_fit(
            self.model, xdata, ydata,
            full_output=True, check_finite=True,
            **self.configuration(**kwargs)
        )
        return {
            "parameters": solution[0],
            "covariance": solution[1],
            "info": solution[2],
            "message": solution[3],
            "status": solution[4]
        }

    def predict(self, xdata):
        if hasattr(self, "_solution"):
            return self.model(xdata, *self._solution["parameters"])
        else:
            raise NotSolvedError("Model must be fitted prior to this operation")

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

    def plot(self, resolution=100):
        xlin = np.linspace(self._xdata.min(), self._xdata.max(), resolution)
        fig, axe = plt.subplots()
        axe.plot(self._xdata, self._ydata, linestyle="none", marker=".", label="Data")
        axe.plot(xlin, self.predict(xlin), label="Fit")
        axe.legend()
        axe.grid()
        return axe

