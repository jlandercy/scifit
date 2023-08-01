"""
Module :py:mod:`scifit.interfaces.generic` defines the class :class:`GenericInterface`
on which any other interfaces must inherit from. This class exposes generic abstract methods
all interfaces must implement.
"""

import abc
from typing import Any

import numpy as np
from scipy import optimize

from scifit.errors.base import *


class FitSolverInterface(abc.ABC):

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
    def model(x, *parameters):
        """
        Model definition to fit with experimental data.
        This method must be overridden by subclassing
        """
        raise MissingModel("Model not defined")

    @classmethod
    def model_score(cls, x, y, *parameters):
        return np.sum(np.power((cls.model(x, *parameters) - y), 2))/y.shape[0]

    def solve(self, xdata, ydata, **kwargs):
        """
        Solve fitting problem and return structured solution
        """
        solution = optimize.curve_fit(
            self.model,
            self._xdata, self._ydata,
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

    def score(self, xdata, ydata):
        if hasattr(self, "_solution"):
            return self.model_score(xdata, ydata, *self._solution["parameters"])
        else:
            raise NotSolvedError("Model must be fitted prior to compute errors")

    def fit(self, xdata, ydata, **kwargs):
        """
        Solve fitting problem and store data
        """
        self.store(xdata, ydata)
        self._solution = self.solve(self._xdata, self._ydata, **kwargs)
        self._yhat = self.model(self._xdata, *self._solution["parameters"])
        self._score = self.score(self._xdata, self._ydata)
        return self._solution

