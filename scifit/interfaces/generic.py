"""
Module :py:mod:`scifit.interfaces.generic` defines the class :class:`GenericInterface`
on which any other interfaces must inherit from. This class exposes generic abstract methods
all interfaces must implement.
"""

import abc
from typing import Any

import numpy as np

from scifit.errors.base import InputDataError, MissingModel


class FitSolver(abc.ABC):
    """
    Generic Interface (Abstract Base Class) for all object of the package.
    This class must be subclassed by any other interfaces.
    """

    def __init__(self, xdata, ydata, *args, **kwargs):

        xdata = np.array(xdata)
        ydata = np.array(ydata)

        if xdata.shape[0] != ydata.shape[0]:
            raise InputDataError("Incompatible shapes between x %s and y %s" % (xdata.shape, ydata.shape))

        self._xdata = xdata
        self._ydata = ydata

    @staticmethod
    def model(x, y, *args):
        raise MissingModel("Model not defined")
