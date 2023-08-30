import inspect
import itertools
import numbers
from collections.abc import Iterable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats

from scifit import logger
from scifit.errors.base import *
from scifit.interfaces.mixins import *
from scifit.toolbox.report import ReportProcessor


class ActivatedStateModelKinetic:
    _names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    _modes = {"direct", "indirect", "equilibrium"}

    def __init__(self, nur, x0, k0, nup=None, k0inv=None, mode="direct"):
        nur, x0, k0, nup, k0inv, mode = self.validate(
            nur, x0, k0, nup=nup, k0inv=k0inv, mode=mode
        )
        self._nur = nur
        self._nup = nup
        self._x0 = x0
        self._k0 = k0
        self._k0inv = k0inv
        self._mode = mode

    @staticmethod
    def split_nus(nus):
        nur = np.copy(nus)
        nur[nur > 0.0] = 0.0
        nup = np.copy(nus)
        nup[nup < 0.0] = 0.0
        return nur, nup

    def validate(self, nur, x0, k0, nup=None, k0inv=None, mode=None):

        # Split nu to make it easier to encode:
        if nup is None and np.any(nur < 0.0):
            nur, nup = self.split_nus(nur)

        nur = np.array(nur)
        nup = np.array(nup)
        x0 = np.array(x0)

        if k0inv is None:
            k0inv = 1.0 / k0

        if mode not in self._modes:
            raise ConfigurationError(
                "Mode must be in %s, git '%s' instead" % (self._modes, mode)
            )

        if nur.ndim != 2:
            raise ConfigurationError("Matrice nur must have two dimensions")

        if nup.ndim != 2:
            raise ConfigurationError("Matrice nur must have two dimensions")

        if nur.shape[1] != x0.shape[0]:
            raise ConfigurationError("Number of columns of nur must equal to x0 size")

        if nup.shape[1] != x0.shape[0]:
            raise ConfigurationError("Number of columns of nup must equal to x0 size")

        if not np.issubdtype(nur.dtype, np.number):
            raise ConfigurationError("Values of nur must be numerical")

        if not np.issubdtype(nup.dtype, np.number):
            raise ConfigurationError("Values of nup must be numerical")

        if not np.all(nur <= 0.0):
            raise ConfigurationError("Values of nur must be negative")

        if not np.all(nup >= 0.0):
            raise ConfigurationError("Values of nup must be positive")

        if not np.issubdtype(x0.dtype, np.number):
            raise ConfigurationError("Values of x0 must be numerical")

        if not np.issubdtype(k0.dtype, np.number):
            raise ConfigurationError("Vector k0 must be numerical")

        if np.all(k0 <= 0.0):
            raise ConfigurationError("Vector k0 must be strictly positive")

        if not np.issubdtype(k0inv.dtype, np.number):
            raise ConfigurationError("Vector k0inv must be numerical")

        if np.all(k0inv <= 0.0):
            raise ConfigurationError("Vector k0inv must be strictly positive")

        return nur, x0, k0, nup, k0inv, mode

    @property
    def n(self):
        return self._nur.shape[0]

    @property
    def k(self):
        return self._nur.shape[1]

    def reactant_indices(self, j):
        return np.where(self._nur[j, :] < 0.0)[0]

    def product_indices(self, j):
        return np.where(self._nup[j, :] > 0.0)[0]

    @property
    def nus(self):
        return self._nur + self._nup

    def model(self, t, x):

        substance_rates = np.full(self._x0.shape, 0.0)

        if self._mode == "direct" or self._mode == "equilibrium":
            reaction_rates = self._k0 * np.prod(np.power(np.row_stack([x]*self.n), np.abs(self._nur)), axis=1)
            substance_rates += np.sum((+self.nus) * np.column_stack([reaction_rates] * self.k), axis=0)

        if self._mode == "indirect" or self._mode == "equilibrium":
            reaction_rates = self._k0inv * np.prod(np.power(np.row_stack([x]*self.n), np.abs(self._nup)), axis=1)
            substance_rates += np.sum((-self.nus) * np.column_stack([reaction_rates] * self.k), axis=0)

        return substance_rates

    def quotient(self, x):
        return np.prod(np.power(np.row_stack([x]*self.n), np.abs(self.nus)), axis=1)

    def equilibrium(self):
        return self._k0 / self._k0inv

    def solve(self, t):
        t = np.array(t)
        tspan = np.array([t.min(), t.max()])
        solution = integrate.solve_ivp(
            self.model, tspan, self._x0, t_eval=t, dense_output=True, atol=1e-14, rtol=1e-8
        )
        self._solution = solution
        self._quotients = np.apply_along_axis(self.quotient, 0, self._solution.y)
        return solution

    def arrow(self, mode="normal"):
        if self._mode == "direct":
            return r" \rightarrow " if mode == "latex" else " -> "
        elif self._mode == "indirect":
            return r" \leftarrow " if mode == "latex" else " <- "
        else:
            return r" \Leftrightarrow " if mode == "latex" else " <=> "

    def model_formula(self, index, mode="normal"):
        formula = " + ".join(
            [
                "{:.2g}{:s}".format(-self.nus[index, k], self._names[k])
                for k in self.reactant_indices(index)
            ]
        )
        formula += self.arrow(mode="latex")
        formula += " + ".join(
            [
                "{:.2g}{:s}".format(self.nus[index, k], self._names[k])
                for k in self.product_indices(index)
            ]
        )
        return formula

    def model_formulas(self, mode="normal"):
        return "; ".join([self.model_formula(j) for j in range(self.n)])

    def plot_solve(self):
        fig, axe = plt.subplots()
        axe.plot(self._solution.t, self._solution.y.T)
        axe.set_title("Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex"))
        axe.set_xlabel("Time, $t$")
        axe.set_ylabel("Concentrations, $x_i$")
        axe.legend(list(self._names[: self.k]))
        #axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)
        return axe

    def plot_quotients(self):
        fig, axe = plt.subplots()
        axe.plot(self._solution.t, self._quotients.T)
        axe.set_title("Activated State Model Kinetic:\nReaction Quotient Evolutions")
        axe.set_xlabel("Time, $t$")
        axe.set_ylabel("Reaction Quotients, $Q_i$")
        axe.legend(["$Q_{%d}$: $%s$" % (i, self.model_formula(i)) for i in range(self.n)])
        #axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)
        return axe
