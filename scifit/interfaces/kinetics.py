import functools
import inspect
import itertools
import numbers
from collections.abc import Iterable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, interpolate, signal

from scifit import logger
from scifit.errors.base import *
from scifit.interfaces.mixins import *
from scifit.toolbox.report import ReportProcessor


class KineticSolverInterface:
    """
    Class solving the Activated State Model Kinetics for several setups
    """

    _names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    _modes = {"direct", "indirect", "equilibrium"}

    def __init__(
        self,
        nur,
        x0,
        k0,
        nup=None,
        k0inv=None,
        mode="direct",
        substance_index=0,
        steady=None,
    ):
        """
        Initialize class with reactions setup

        :param nur:
        :param x0:
        :param k0:
        :param nup:
        :param k0inv:
        :param mode:
        """
        nur, x0, k0, nup, k0inv, mode = self.validate(
            nur, x0, k0, nup=nup, k0inv=k0inv, mode=mode
        )

        self._substance_index = substance_index

        self._nur = nur
        self._nup = nup
        self._x0 = x0
        self._k0 = k0
        self._k0inv = k0inv
        self._mode = mode

        if steady is None:
            steady = np.array([1.0] * self.k)
        self._steady = steady

    @staticmethod
    def split_nus(nus):
        """
        Split stoechiometric coefficients matrix into reactants and products

        :param nus:
        :return:
        """

        nur = np.copy(nus)
        nur[nur > 0.0] = 0.0

        nup = np.copy(nus)
        nup[nup < 0.0] = 0.0

        return nur, nup

    def validate(self, nur, x0, k0, nup=None, k0inv=None, mode=None):
        """
        Validate and sanitize user input

        :param nur:
        :param x0:
        :param k0:
        :param nup:
        :param k0inv:
        :param mode:
        :return:
        """

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
        """
        Return the number of reactions

        :return:
        """
        return self._nur.shape[0]

    @property
    def k(self):
        """
        Return the number of substances envolved into reactions

        :return:
        """
        return self._nur.shape[1]

    def reactant_indices(self, reaction_index):
        """
        Return reactant indices for a given reaction

        :param reaction_index:
        :return:
        """
        return np.where(self._nur[reaction_index, :] < 0.0)[0]

    def product_indices(self, reaction_index):
        """
        Return product indices for a given reaction

        :param reaction_index:
        :return:
        """
        return np.where(self._nup[reaction_index, :] > 0.0)[0]

    @property
    def references(self):
        return np.array([self.reactant_indices(i)[0] for i in range(self.n)])

    @property
    def nus(self):
        """
        Return the complete stoechiometric coefficient matrix

        :return:
        """
        return self._nur + self._nup

    @property
    def direct_orders(self):
        """
        Return direct reaction total orders

        :return:
        """
        return np.sum(self._nur, axis=1)

    @property
    def indirect_orders(self):
        """
        Return indirect reaction total orders

        :return:
        """
        return np.sum(self._nup, axis=1)

    def model(self, t, x, k0, k0inv):
        """
        Compute reaction rate for each reaction, then compute substance rates
        in order to solve the ODE system of the kinetic.

        Where each global reaction rate is defined as follows:

        .. math::

            r_i^{\\rightarrow} = \\frac{1}{V}\\frac{\\partial \\xi}{\\partial t} = k_i^{\\rightarrow} \\cdot \\prod\\limits_{j=1}^{j=k} x_j^{|\\nu_{i,j}^R|} \\, , \\quad \\forall i \\in \\{1,\\dots, n\\}

        .. math::

            -r_i^{\\leftarrow} = \\frac{1}{V}\\frac{\\partial \\xi}{\\partial t} = k_i^{\\leftarrow} \\cdot \\prod\\limits_{j=1}^{j=k} x_j^{|\\nu_{i,j}^P|} \\, , \\quad \\forall i \\in \\{1,\\dots, n\\}


        And each substance reaction rate is defined as:

        .. math::

            r_j^{\\rightarrow} = \\sum\\limits_{i=1}^{i=n} \\nu_{i,j} \\cdot r_i \\, , \\quad \\forall j \\in \\{1,\\dots, k\\}

        .. math::

            r_j^{\\leftarrow} = \\sum\\limits_{i=1}^{i=n} -\\nu_{i,j} \\cdot r_i \\, , \\quad \\forall j \\in \\{1,\\dots, k\\}

        .. math::

            r_j^{\\leftrightharpoons} = r_j^{\\leftarrow} + r_j^{\\rightarrow} \\, , \\quad \\forall j \\in \\{1,\\dots, k\\}

        :param t:
        :param x:
        :return:
        """

        substance_rates = np.full(self._x0.shape, 0.0)

        if self._mode == "direct" or self._mode == "equilibrium":
            reaction_rates = k0 * np.prod(
                np.power(np.row_stack([x] * self.n), np.abs(self._nur)), axis=1
            )
            # reaction_rates = k0 * np.power(10., np.sum(
            #     np.log10(np.row_stack([x] * self.n) * np.abs(self._nur)), axis=1
            # ))
            substance_rates += np.sum(
                (+self.nus) * np.column_stack([reaction_rates] * self.k), axis=0
            )

        if self._mode == "indirect" or self._mode == "equilibrium":
            reaction_rates = k0inv * np.prod(
                np.power(np.row_stack([x] * self.n), np.abs(self._nup)), axis=1
            )
            substance_rates += np.sum(
                (-self.nus) * np.column_stack([reaction_rates] * self.k), axis=0
            )

        return (
            np.round(substance_rates, np.finfo(np.longdouble).precision) * self._steady
        )

    def parametered_model(self, k0, k0inv):
        def wrapped(t, x):
            return self.model(t, x, k0=k0, k0inv=k0inv)

        return wrapped

    def solve(self, t, k0, k0inv):
        t = np.array(t)
        tspan = np.array([t.min(), t.max()])
        model = self.parametered_model(k0, k0inv)
        solution = integrate.solve_ivp(
            model,
            tspan,
            self._x0,
            t_eval=t,
            dense_output=True,
            method="LSODA",
            min_step=1e-8,
            atol=1e-12,
            rtol=1e-10,
        )
        return solution

    def fit(self, t, k0=None, k0inv=None):
        """
        Solve the ODE system defined as follows:

        :param t:
        :return:
        """

        if k0 is None:
            k0 = self._k0

        if k0inv is None:
            k0inv = self._k0inv

        self._solution = self.solve(t, k0, k0inv)
        self._quotients = np.apply_along_axis(self.quotient, 0, self._solution.y)
        self._dxdt = self.derivative(derivative_order=1)
        self._d2xdt2 = self.derivative(derivative_order=2)
        self._selectivities = self.selectivities()
        self._levenspiel = self.levenspiel()
        return self._solution

    def quotient(self, x):
        """
        Return Reaction quotient for each reaction at the given concentration

          .. math::

            Q_i = \\prod\\limits_{j=1}^{j=k} x_j^{\\nu_{i,j}} \\, , \\quad \\forall i \\in \\{1,\\dots, n\\}

        Computed as:

        .. math::

            Q_i = 10^{\\sum\\limits_{j=1}^{j=k} \\log_{10}(x_j) \\cdot \\nu_{i,j}} \\, , \\quad \\forall i \\in \\{1,\\dots, n\\}

        :param x:
        :return:
        """
        # return np.prod(np.power(np.row_stack([x] * self.n), self.nus), axis=1)
        return np.power(
            10, np.sum(np.log10(np.row_stack([x] * self.n)) * self.nus, axis=1)
        )

    def equilibrium_constants(self):
        """
        Return Reaction Equilibrium constant for each reaction

        .. math::

            K_i^\\leftrightharpoons = \\frac{k_i^\\rightarrow}{k_i^\\leftarrow} \\, , \\quad \\forall i \\in \\{1,\\dots, n\\}

        :return:
        """
        return self._k0 / self._k0inv

    def convertion_ratio(self, substance_index=None):
        """
        Return the conversion ration of a given substance

        :param substance_index:
        :return:
        """
        substance_index = substance_index or self._substance_index or 0
        x = self._solution.y.T
        x0 = self._x0[substance_index]
        return (x0 - x[:, substance_index]) / x0

    def derivative(self, derivative_order=1, polynomial_order=3, window=21):
        """
        Return the n-th derivative of a kinetic using Savitsky-Golay filter for estimation

        :param derivative_order:
        :param polynomial_order:
        :param window:
        :return:
        """
        dxdt = signal.savgol_filter(
            self._solution.y.T,
            window_length=window,
            polyorder=polynomial_order,
            deriv=derivative_order,
            delta=np.diff(self._solution.t)[0],
            axis=0,
            mode="interp",
        )
        dxdt = np.round(dxdt, np.finfo(np.longdouble).precision)
        return dxdt

    def selectivities(self, substance_index=None):
        """
        Return instantaneous selectivities using concentration first derivative estimates

        .. math::

            \\mathcal{S}_{i,j} = \\frac{\\frac{\\partial x_j}{\\partial t}}{\\frac{\\partial x_i}{\\partial t}}

        :param substance_index:
        :return:
        """
        substance_index = substance_index or self._substance_index or 0
        dxdt = self.derivative(derivative_order=1)
        selectivities = (dxdt.T / (dxdt[:, substance_index])).T
        return selectivities

    def levenspiel(self, substance_index=None):
        """
        Return Levenspiel integration using concentration first derivative estimates

        :param substance_index:
        :return:
        """
        substance_index = substance_index or self._substance_index or 0
        L = 1.0 / (
            np.abs(self.derivative(derivative_order=1)) + np.finfo(np.longdouble).eps
        )
        x0 = self._solution.y.T[:, substance_index]
        I = integrate.cumulative_trapezoid(L, x0, axis=0)
        return I

    def arrow(self, mode="normal"):
        """
        Generate arrow for reactions

        :param mode:
        :return:
        """
        if self._mode == "direct":
            return r" \rightarrow " if mode == "latex" else " -> "
        elif self._mode == "indirect":
            return r" \leftarrow " if mode == "latex" else " <- "
        else:
            return r" \leftrightharpoons " if mode == "latex" else " <=> "

    def model_formula(self, index, mode="normal"):
        """
        Generate reaction formula

        :param index:
        :param mode:
        :return:
        """
        formula = " + ".join(
            [
                "{:.2g}{:s}".format(-self._nur[index, k], self._names[k])
                for k in self.reactant_indices(index)
            ]
        )
        formula += self.arrow(mode="latex")
        formula += " + ".join(
            [
                "{:.2g}{:s}".format(self._nup[index, k], self._names[k])
                for k in self.product_indices(index)
            ]
        )
        return formula

    def model_formulas(self, mode="normal"):
        """
        Generate reaction formulas

        :param mode:
        :return:
        """
        return "; ".join([self.model_formula(j) for j in range(self.n)])

    def dataset(self):
        data = pd.DataFrame({"t": self._solution.t})
        concentrations = pd.DataFrame(
            self._solution.y.T, columns=[self._names[i] for i in range(self.k)]
        )
        derivatives = pd.DataFrame(
            self._dxdt, columns=["d%s/dt" % self._names[i] for i in range(self.k)]
        )
        quotients = pd.DataFrame(
            self._quotients.T, columns=["Q%d" % i for i in range(self.n)]
        )
        data = pd.concat([data, concentrations, derivatives, quotients], axis=1)
        return data

    def plot_solve(self, substance_indices=None):
        """
        Plot ODE solution figure

        :return:
        """

        if substance_indices is None:
            substance_indices = np.arange(self.k)

        fig, axe = plt.subplots()
        axe.plot(self._solution.t, self._solution.y.T[:, substance_indices])
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        axe.set_xlabel("Time, $t$")
        axe.set_ylabel("Concentrations, $x_i$")
        axe.legend(list(self._names[: self.k]))
        # axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def plot_solve_ratio(self, substance_indices=None):
        """
        Plot ODE solution figure wrt to conversion ratio

        :return:
        """

        if substance_indices is None:
            substance_indices = np.arange(self.k)

        r = self.convertion_ratio()
        fig, axe = plt.subplots()
        axe.plot(r, self._solution.y.T[:, substance_indices])
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        axe.set_xlabel(r"Conversion Ratio, $\rho$")
        axe.set_ylabel("Concentrations, $x_i$")
        axe.legend(list(self._names[: self.k]))
        # axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def plot_first_derivative(self, substance_indices=None):
        """
        Plot ODE solution first derivative figure

        :return:
        """

        if substance_indices is None:
            substance_indices = np.arange(self.k)

        fig, axe = plt.subplots()
        axe.plot(self._solution.t, self._dxdt[:, substance_indices])
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        axe.set_xlabel("Time, $t$")
        axe.set_ylabel("Concentration Velocities, $\partial x_i / \partial t$")
        axe.legend(list(self._names[: self.k]))
        # axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def plot_second_derivative(self, substance_indices=None):
        """
        Plot ODE solution first derivative figure

        :return:
        """

        if substance_indices is None:
            substance_indices = np.arange(self.k)

        fig, axe = plt.subplots()
        axe.plot(self._solution.t, self._d2xdt2[:, substance_indices])
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        axe.set_xlabel("Time, $t$")
        axe.set_ylabel("Concentration Accelerations, $\partial^2 x_i / \partial t^2$")
        axe.legend(list(self._names[: self.k]))
        # axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def plot_selectivities(self, substance_indices=None):
        """
        Plot ODE solution selectivities figure

        :return:
        """

        if substance_indices is None:
            substance_indices = np.arange(self.k)

        fig, axe = plt.subplots()
        axe.plot(self._solution.t, self._selectivities[:, substance_indices])
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        axe.set_xlabel("Time, $t$")
        axe.set_ylabel("Instataneous Selectivities, $S_i$")
        axe.legend(list(self._names[: self.k]))
        # axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def plot_levenspiel(self, substance_indices=None):
        """
        Plot ODE solution Levenspiel integration figure

        :return:
        """

        if substance_indices is None:
            substance_indices = np.arange(self.k)

        fig, axe = plt.subplots()
        axe.plot(self._solution.t[:-1], self._levenspiel[:, substance_indices])
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        axe.set_xlabel("Time, $t$")
        axe.set_ylabel("Levenspiel Integral, $L_i$")
        axe.legend(list(self._names[: self.k]))
        # axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def plot_quotients(self):
        """
        Plot the reaction quotient for the solved system

        :return:
        """

        fig, axe = plt.subplots()
        for i, Q in enumerate(self._quotients):
            axe.plot(
                self._solution.t, Q, label="$Q_{%d}$: $%s$" % (i, self.model_formula(i))
            )

        if self._mode == "equilibrium":
            for i, K in enumerate(self.equilibrium_constants()):
                axe.axhline(K, linestyle="-.", color="black", label=r"$K_{%d}$" % i)

        axe.set_title("Activated State Model Kinetic:\nReaction Quotient Evolutions")
        axe.set_xlabel("Time, $t$")
        axe.set_ylabel("Reaction Quotients, $Q_i$")
        axe.legend()
        axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe
