import inspect
import itertools
import numbers
from collections.abc import Iterable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, integrate, stats

from scifit import logger
from scifit.errors.base import *
from scifit.interfaces.mixins import *
from scifit.toolbox.report import ReportProcessor


class ActivatedStateModelKinetic:

    _names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, nus, x0, k0):
        nus, x0, k0 = self.validate(nus, x0, k0)
        self._nus = nus
        self._x0 = x0
        self._k0 = k0

    def validate(self, nus, x0, k0):

        nus = np.array(nus)
        x0 = np.array(x0)

        if nus.shape != x0.shape:
            raise ConfigurationError("Shape of nus and x0 must be identical")

        if not np.issubdtype(nus.dtype, np.number):
            raise ConfigurationError("Values of nus must be numerical")

        if not np.issubdtype(x0.dtype, np.number):
            raise ConfigurationError("Values of x0 must be numerical")

        if not isinstance(k0, numbers.Number):
            raise ConfigurationError("Constant k0 must be numerical")

        if k0 <= 0.:
            raise ConfigurationError("Constant k0 must be strictly positive")

        return nus, x0, k0

    @property
    def n(self):
        return self._nus.shape[0]

    def model(self, t, x):
        reactants = np.where(self._nus < 0.)[0]
        rate = self._k0*np.prod(np.power(x[reactants], np.abs(self._nus[reactants])))
        rates = self._nus * np.full(self._nus.shape, rate)
        return rates

    def solve(self, t):
        t = np.array(t)
        tspan = np.array([t.min(), t.max()])
        solution = integrate.solve_ivp(self.model, tspan, self._x0, t_eval=t, dense_output=True)
        self._solution = solution
        return solution

    def plot_solve(self):
        fig, axe = plt.subplots()
        axe.plot(self._solution.t, self._solution.y.T)
        axe.set_title("Activated State Model Kinetic")
        axe.set_xlabel("Time, $t$")
        axe.set_ylabel("Concentrations, $x_i$")
        axe.legend(list(self._names[:self.n]))
        axe.grid()
        fig.subplots_adjust(top=0.9, left=0.2)
        return axe
