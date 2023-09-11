import functools
import inspect
import itertools
import numbers
import warnings
from collections.abc import Iterable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, interpolate, signal

from scifit import logger
from scifit.errors.base import *
from scifit.interfaces.mixins import *
from scifit.toolbox.report import KineticSolverReportProcessor


class KineticSolverInterface:
    """
    Class solving the Activated State Model Kinetics for several setups
    """

    _precision = np.finfo(np.longdouble).precision
    _eps = np.finfo(np.longdouble).eps

    _names = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
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
        unsteady=None,
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

        # Split nu to make it easier to encode:
        if nup is None and np.any(nur < 0.0):
            nur, nup = self.split_nus(nur)

        if unsteady is None:
            unsteady = np.full((nur.shape[1],), True)
        unsteady = unsteady.astype(bool).astype(float)

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

        if not np.issubdtype(unsteady.dtype, np.number):
            raise ConfigurationError("Vector unsteady must be numerical")

        if unsteady.shape != x0.shape:
            raise ConfigurationError("Vector unsteady must have the same shape than x0")

        if not isinstance(substance_index, numbers.Integral):
            raise ConfigurationError("Substance index must be an integer")

        if not (0 <= substance_index < nur.shape[1]):
            raise ConfigurationError(
                "Substance index must be in {0, ..., %d}" % nur.shape[1]
            )

        self._substance_index = substance_index
        self._nur = nur
        self._nup = nup
        self._x0 = x0
        self._k0 = k0
        self._k0inv = k0inv
        self._mode = mode
        self._unsteady = unsteady

    # @classmethod
    # def threshold(cls, x, eps=None):
    #     """
    #     Apply threshold (default is machine precsion)
    #
    #     :param x:
    #     :param eps:
    #     :return:
    #     """
    #     eps = eps or cls._eps
    #     x[np.abs(x) <= eps] = eps
    #     return x

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

    @property
    def reaction_space_size(self):
        """Number of reactions in the system"""
        return self._nur.shape[0]

    @property
    def substance_space_size(self):
        """Number of substances envolved into the system"""
        return self._nur.shape[1]

    @property
    def n(self):
        """
        Number of reactions in the system

        :return:
        """
        return self.reaction_space_size

    @property
    def k(self):
        """
        Number of substances envolved into the system

        :return:
        """
        return self.substance_space_size

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
        """
        Reaction reference for each substance, find the first reaction involving the substance
        :return:
        """
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

    def system(self, t, x, k0=None, k0inv=None, x0=None):
        """
        Compute reaction rate for each reaction, then compute substance rates
        in order to solve the ODE system of the kinetic.

        Where each global reaction rate is defined as follows:

        .. math::

            r_i^{\\rightarrow} = \\frac{1}{V}\\frac{\\partial \\xi_i^{\\rightarrow}}{\\partial t} = k_i^{\\rightarrow} \\cdot \\prod\\limits_{j=1}^{j=k} x_j^{|\\nu_{i,j}^R|} \\, , \\quad \\forall i \\in \\{1,\\dots, n\\}

        .. math::

            r_i^{\\leftarrow} = \\frac{1}{V}\\frac{\\partial \\xi_i^{\\leftarrow}}{\\partial t} = k_i^{\\leftarrow} \\cdot \\prod\\limits_{j=1}^{j=k} x_j^{|\\nu_{i,j}^P|} \\, , \\quad \\forall i \\in \\{1,\\dots, n\\}


        And each substance reaction rate is defined as:

        .. math::

            R_j^{\\rightarrow} = \\frac{\\partial x_j^{\\rightarrow}}{\\partial t} = \\sum\\limits_{i=1}^{i=n} \\nu_{i,j} \\cdot r_i^{\\rightarrow} \\, , \\quad \\forall j \\in \\{1,\\dots, k\\}

        .. math::

            R_j^{\\leftarrow} = \\frac{\\partial x_j^{\\leftarrow}}{\\partial t} = - \\sum\\limits_{i=1}^{i=n} \\nu_{i,j} \\cdot r_i^{\\leftarrow} \\, , \\quad \\forall j \\in \\{1,\\dots, k\\}

        .. math::

            R_j^{\\leftrightharpoons} = \\frac{\\partial x_j^{\\leftrightharpoons}}{\\partial t} = R_j^{\\rightarrow} - R_j^{\\leftarrow} \\, , \\quad \\forall j \\in \\{1,\\dots, k\\}

        :param t:
        :param x:
        :return:
        """

        if x0 is None:
            x0 = self._x0

        if k0 is None:
            k0 = self._k0

        if k0inv is None:
            k0inv = self._k0inv

        substance_rates = np.full(x0.shape, 0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

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

        # return (
        #     np.round(substance_rates, np.finfo(np.longdouble).precision) * self._steady
        # )
        return substance_rates * self._unsteady

    def parametered_system(self, k0, k0inv):
        """
        Wrapper: System parametered exposing only time and concentrations
        :param k0:
        :param k0inv:
        :return:
        """

        def wrapped(t, x, x0=None):
            return self.system(t, x, k0=k0, k0inv=k0inv, x0=x0)

        return wrapped

    def time_parametered_system(self, t, k0, k0inv):
        """
        Wrapper: System parametered exposing only time

        :param t:
        :param k0:
        :param k0inv:
        :return:
        """

        def wrapped(x, x0=None):
            return self.system(t, x, k0=k0, k0inv=k0inv, x0=x0)

        return wrapped

    def rates(self):
        """
        Evaluate system to get accurate rates
        :return:
        """
        model = self.time_parametered_system(self._solution.t, self._k0, self._k0inv)
        dxdt = np.apply_along_axis(model, 1, self._solution.y.T)
        return dxdt

    def accelerations(self):
        """
        Differentiate rates to get accelerations

        :return:
        """
        return self.derivative(data=self.rates())

    def integrate_system(self, t, k0=None, k0inv=None, x0=None):
        """
        Integrate system with specific conditioning to cope with stiffness.

        System is defined as follow:

        .. math::

            \\frac{\\partial x_j}{\\partial t} = \\sum\\limits_{i=0}^{i=n} \\nu_{i,j} \\cdot r_i \\, , \\quad \\forall j \\in \\{1, \\dots, k \\}

        :param t:
        :param k0:
        :param k0inv:
        :param x0:
        :return:
        """
        t = np.array(t)
        tspan = np.array([t.min(), t.max()])

        if x0 is None:
            x0 = self._x0

        system = self.parametered_system(k0=k0, k0inv=k0inv)

        solution = integrate.solve_ivp(
            system,
            tspan,
            x0,
            t_eval=t,
            dense_output=True,
            method="LSODA",
            min_step=1e-8,
            atol=1e-12,
            rtol=1e-10,
        )

        return solution

    def integrate(self, t, k0=None, k0inv=None, x0=None, substance_index=None):
        """
        Solve the ODE system and compute extra aggregates

        :param t:
        :return:
        """

        substance_index = substance_index or self._substance_index or 0

        self._solution = self.integrate_system(t, k0=k0, k0inv=k0inv, x0=x0)
        self._quotients = np.apply_along_axis(self.quotient, 0, self._solution.y)
        self._rates = self.rates()
        self._accelerations = self.accelerations()
        self._selectivities = self.selectivities(substance_index=substance_index)
        self._global_selectivities = self.global_selectivities(
            substance_index=substance_index
        )
        self._yields = self.yields(substance_index=substance_index)
        self._levenspiel = self.levenspiel()
        self._integrated_levenspiel = self.integrated_levenspiel()
        return self._solution

    def quotient(self, x):
        """
        Return Reaction quotient for each reaction at the given concentration

        .. math::

            Q_i = \\prod\\limits_{j=1}^{j=k} x_j^{\\nu_{i,j}} \\, , \\quad \\forall i \\in \\{1,\\dots, n\\}

        Computed as:

        .. math::

            Q_i = 10^{\\sum\\limits_{j=1}^{j=k} \\nu_{i,j} \\cdot \\log_{10}(x_j)} \\, , \\quad \\forall i \\in \\{1,\\dots, n\\}

        :param x:
        :return:
        """
        # return np.prod(np.power(np.row_stack([x] * self.n), self.nus), axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

    def derivative(
        self, data=None, derivative_order=1, polynomial_order=3, window=21, rounded=True
    ):
        """
        Return the n-th derivative of a kinetic using Savitsky-Golay filter for estimation

        :param data:
        :param derivative_order:
        :param polynomial_order:
        :param window:
        :return:
        """

        if data is None:
            data = self._solution.y.T

        dxdt = signal.savgol_filter(
            data,
            window_length=window,
            polyorder=polynomial_order,
            deriv=derivative_order,
            delta=np.diff(self._solution.t)[0],
            axis=0,
            mode="interp",
        )

        # Drastically reduce noise on selectivity:
        if rounded:
            dxdt = np.round(dxdt, self._precision)
            # dxdt = self.threshold(dxdt)

        # def factory(t, n=derivative_order):
        #     def wrapped(x):
        #         return interpolate.UnivariateSpline(t, x, k=5, s=6).derivative(n=n)(t)
        #     return wrapped
        #
        # dxdt = np.apply_along_axis(factory(self._solution.t), 0, data)

        return dxdt

    def quotient_rates(self):
        """
        Compute quotient rates
        :return:
        """
        return self.derivative(data=self._quotients.T)

    def selectivities(self, substance_index=None):
        """
        Return instantaneous selectivities using concentration first derivative estimates

        .. math::

            \\mathcal{S}_{r,j} = \\frac{\\frac{\\partial x_j}{\\partial t}}{\\frac{\\partial x_r}{\\partial t}}  \\, , \\quad \\forall r, j \\in \\{1,\\dots, k\\}

        :param substance_index:
        :return:
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            substance_index = substance_index or self._substance_index or 0
            dxdt = self.derivative(derivative_order=1)
            selectivities = (dxdt.T / (dxdt[:, substance_index])).T
            # selectivities = self.derivative(data=self.integrated_selectivities(substance_index=substance_index))
        return np.round(selectivities, np.finfo(np.longdouble).precision)

    def global_selectivities(self, substance_index=None):
        """
        Return global selectivities using instantaneous selectivity estimates

        .. math::

            S_{r,j} = \\frac{\\int\\limits_{x_{r,0}}^{x_r} \\mathcal{S}_{r,j} \\cdot \\mathrm{d}x_r}{\\int\\limits_{x_{r,0}}^{x_r} \\mathrm{d}x_r} \\, , \\quad \\forall r, j \\in \\{1,\\dots, k\\}

        :param substance_index:
        :return:
        """
        substance_index = substance_index or self._substance_index or 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S = self.selectivities(substance_index=substance_index)
            x0 = self._solution.y.T[:, substance_index]
            I = integrate.cumulative_trapezoid(S, x0, axis=0, initial=0.0)
            scaler = integrate.cumulative_trapezoid(
                np.full(x0.shape, 1.0), x0, axis=0, initial=0.0
            )
            I = (I.T / scaler).T
        return I

    def yields(self, substance_index=None):
        """
        Return yields using integrated selectivities

        :param substance_index:
        :return:
        """
        substance_index = substance_index or self._substance_index or 0

        return (self._solution.y.T - self._x0) / self._x0[substance_index]

    def levenspiel(self):
        """
        Return Levenspiel curve using concentration first derivative estimates

        :param substance_index:
        :return:
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            L = 1.0 / (self.derivative(derivative_order=1))
        return L

    def integrated_levenspiel(self, substance_index=None):
        """
        Return Levenspiel integration using concentration first derivative estimates

        :param substance_index:
        :return:
        """
        substance_index = substance_index or self._substance_index or 0
        L = self.levenspiel()
        x0 = self._solution.y.T[:, substance_index]
        I = integrate.cumulative_trapezoid(L, x0, axis=0)
        return I

    def arrow(self, mode="normal", overset=None):
        """
        Generate arrow for reactions

        :param mode:
        :return:
        """
        if self._mode == "direct":
            if mode == "normal":
                return " -> "
            elif mode == "latex":
                if overset:
                    return r" \overset{%s}{\rightarrow} " % str(overset)
                else:
                    return r" \rightarrow "
        elif self._mode == "indirect":
            if mode == "normal":
                return " <- "
            elif mode == "latex":
                if overset:
                    return r" \overset{%s}{\leftarrow} " % str(overset)
                else:
                    return r" \leftarrow "
        elif self._mode == "equilibrium":
            if mode == "normal":
                return " <=> "
            elif mode == "latex":
                if overset:
                    return r" \overset{%s}{\leftrightharpoons} " % str(overset)
                else:
                    return r" \leftrightharpoons "
        else:
            raise ConfigurationError("Bad arrow configuration")

    def model_formula(self, index, mode="normal", overset=None, array=False):
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
        if array:
            formula += " & "
        formula += self.arrow(mode=mode, overset=overset)
        if array:
            formula += " & "
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
        return "; ".join([self.model_formula(j, mode=mode) for j in range(self.n)])

    def model_equations(self):
        """
        Generate reaction formulas

        :param mode:
        :return:
        """
        latex = r"\begin{eqnarray}" + "\n"
        latex += (r" \\" + "\n").join(
            [
                self.model_formula(
                    j, mode="latex", overset=r"\beta_{%d}" % j, array=True
                )
                for j in range(self.n)
            ]
        )
        latex += "\n" + r"\end{eqnarray}" + "\n"
        return latex

    def dataset(self):
        data = pd.DataFrame({"t": self._solution.t})
        concentrations = pd.DataFrame(
            self._solution.y.T, columns=[self._names[i] for i in range(self.k)]
        )
        derivatives = pd.DataFrame(
            self._rates, columns=["d%s/dt" % self._names[i] for i in range(self.k)]
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
        axe.legend(list(self._names[substance_indices]))
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
        axe.legend(list(self._names[substance_indices]))
        # axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def plot_rates(self, substance_indices=None):
        """
        Plot ODE solution first derivative figure

        :return:
        """

        if substance_indices is None:
            substance_indices = np.arange(self.k)

        fig, axe = plt.subplots()
        axe.plot(self._solution.t, self._rates[:, substance_indices])
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        axe.set_xlabel("Time, $t$")
        axe.set_ylabel("Concentration Rates, $\partial x_i / \partial t$")
        axe.legend(list(self._names[substance_indices]))
        # axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def plot_accelerations(self, substance_indices=None):
        """
        Plot ODE solution first derivative figure

        :return:
        """

        if substance_indices is None:
            substance_indices = np.arange(self.k)

        fig, axe = plt.subplots()
        axe.plot(self._solution.t, self._accelerations[:, substance_indices])
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        axe.set_xlabel("Time, $t$")
        axe.set_ylabel("Concentration Accelerations, $\partial^2 x_i / \partial t^2$")
        axe.legend(list(self._names[substance_indices]))
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
        axe.plot(self.convertion_ratio(), self._selectivities[:, substance_indices])
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        axe.set_xlabel(r"Conversion Ratio, $\rho$")
        axe.set_ylabel("Instataneous Selectivities, $\mathcal{S}_i$")
        axe.legend(list(self._names[substance_indices]))
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def plot_global_selectivities(self, substance_indices=None):
        """
        Plot ODE solution selectivities figure

        :return:
        """

        if substance_indices is None:
            substance_indices = np.arange(self.k)

        fig, axe = plt.subplots()
        axe.plot(
            self.convertion_ratio(),
            self._global_selectivities[:, substance_indices],
        )
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        axe.set_xlabel(r"Conversion Ratio, $\rho$")
        axe.set_ylabel("Global Selectivities, $S_i$")
        axe.legend(list(self._names[substance_indices]))
        axe.grid()

        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def plot_yields(self, substance_indices=None):
        """
        Plot ODE solution yields figure

        :return:
        """

        if substance_indices is None:
            substance_indices = np.arange(self.k)

        fig, axe = plt.subplots()
        axe.plot(self.convertion_ratio(), self._yields[:, substance_indices])
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        axe.set_xlabel(r"Conversion Ratio, $\rho$")
        axe.set_ylabel(r"Yields, $y_i$")
        axe.legend(list(self._names[substance_indices]))
        # axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def plot_levenspiel(self, substance_index=None, substance_indices=None):
        """
        Plot ODE solution Levenspiel curve figure

        :return:
        """

        if substance_indices is None:
            substance_indices = np.arange(self.k)

        fig, axe = plt.subplots()
        axe.plot(
            # self._solution.y.T[:, substance_index],
            self.convertion_ratio(substance_index=substance_index),
            np.abs(self._levenspiel[:, substance_indices]),
        )
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        # axe.set_xlabel(r"Reference Concentration, $x_r$")
        axe.set_xlabel(r"Conversion Ratio, $\rho$")
        axe.set_ylabel(r"Levenspiel Curves, $|L_i| = |\frac{1}{r_i}|$")
        axe.legend(list(self._names[substance_indices]))
        axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def plot_integrated_levenspiel(self, substance_indices=None):
        """
        Plot ODE solution Levenspiel integration figure

        :return:
        """

        if substance_indices is None:
            substance_indices = np.arange(self.k)

        fig, axe = plt.subplots()
        axe.plot(
            self.convertion_ratio()[:-1],
            np.abs(self._integrated_levenspiel[:, substance_indices]),
        )
        axe.set_title(
            "Activated State Model Kinetic:\n$%s$" % self.model_formulas(mode="latex")
        )
        axe.set_xlabel(r"Conversion Ratio, $\rho$")
        axe.set_ylabel(r"Levenspiel Integral, $|\int_0^\rho L_i \mathrm{d}\rho|$")
        axe.legend(list(self._names[substance_indices]))
        axe.set_yscale("log")
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
                self._solution.t,
                Q,
                label="$Q_{%d}$: $%s$" % (i, self.model_formula(i, mode="latex")),
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

    def plot_quotient_rates(self):
        """
        Plot the reaction quotient for the solved system

        :return:
        """

        fig, axe = plt.subplots()
        dQ = self.quotient_rates().T
        for i, dQi in enumerate(dQ):
            axe.plot(
                self._solution.t,
                np.abs(dQi),
                label="$\partial Q_{%d}$: $%s$"
                % (i, self.model_formula(i, mode="latex")),
            )

        axe.set_title("Activated State Model Kinetic:\nReaction Quotient Rates")
        axe.set_xlabel("Time, $t$")
        axe.set_ylabel("Quotients Rate, $|\partial Q_i / \partial t|$")
        axe.legend()
        axe.set_yscale("log")
        axe.grid()
        fig.subplots_adjust(top=0.85, left=0.2)

        return axe

    def coefficients(self):
        data = pd.DataFrame(self.nus, columns=self._names[: self.k]).reset_index()
        data["index"] = data["index"].apply(lambda x: "$R_{%d}$" % x)
        data = data.rename(columns={"index": ""})
        return data

    def concentrations(self):
        data = pd.DataFrame({"x0": self._x0, "steady": (1.0 - self._unsteady) == 1}).T
        data.columns = list(self._names[: self.k])
        data = data.reset_index().rename(columns={"index": ""})
        return data

    def constants(self):
        data = pd.DataFrame({"k0": self._k0, "k0inv": self._k0inv})
        data.index = data.index.map(lambda x: "$R_{%d}$" % x)
        data = data.reset_index().rename(columns={"index": ""})
        if self._mode == "direct":
            data.pop("k0inv")
        if self._mode == "indirect":
            data.pop("k0")
        return data

    def report(self, file, path=".", mode="pdf", **kwargs):
        processor = KineticSolverReportProcessor()
        processor.report(self, file=file, path=path, mode=mode, **kwargs)
