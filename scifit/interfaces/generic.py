"""
Module :py:mod:`scifit.interfaces.generic` defines the class :class:`GenericInterface`
on which any other interfaces must inherit from. This class exposes generic abstract methods
all interfaces must implement.
"""

from collections.abc import Iterable
import inspect
import itertools

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
        """
        Fitted mean the fitting procedure has been executed successfully
        """
        is_fitted = hasattr(self, "_solution")
        if not(is_fitted) and error:
            NotFittedError("Model must be fitted prior to this operation")
        return is_fitted

    def solved(self, error=False):
        """
        Solved mean the fitting procedure has been executed successfully and has converged to a potential solution
        """
        has_converged = self.fitted(error=error) and self._solution["status"] in {1, 2, 3, 4}
        if not(has_converged) and error:
            NotSolvedError(
                "Fitting procedure has not converged ({status:}): {message:}".format(**self._solution)
            )
        return has_converged

    def predict(self, xdata, parameters=None):
        if parameters is not None or self.fitted(error=True):
            return self.model(xdata, *(parameters or self._solution["parameters"]))

    def loss(self, xdata, ydata, parameters=None):
        return np.sum(np.power(ydata - self.predict(xdata, parameters=parameters), 2)) / ydata.shape[0]
    loss.name = "MSE"

    def score(self, xdata, ydata, parameters=None):
        RSS = np.sum(np.power(ydata - self.predict(xdata, parameters=parameters), 2))
        TSS = np.sum(np.power(ydata - self._ydata.mean(), 2))
        return 1 - RSS/TSS
    score.name = "$R^2$"

    def parametrized_loss(self):
        """
        Vectorized loss wrt to parameter space
        """
        @np.vectorize
        def wrapper(*parameters):
            return self.loss(self._xdata, self._ydata, parameters=parameters)
        return wrapper

    def fit(self, xdata, ydata, **kwargs):
        """
        Solve fitting problem and store data and results
        """
        self.store(xdata, ydata)
        self._solution = self.solve(self._xdata, self._ydata, **kwargs)
        self._yhat = self.predict(self._xdata)
        self._loss = self.loss(self._xdata, self._ydata)
        self._score = self.score(self._xdata, self._ydata)
        return self._solution

    @staticmethod
    def scale(mode="lin", xmin=0., xmax=1., resolution=101):
        """
        Generate 1-dimensional scale
        """
        if mode == "lin":
            return np.linspace(xmin, xmax, resolution)
        elif mode == "log":
            return np.logspace(xmin, xmax, resolution, base=10)
        else:
            raise ConfigurationError("Domain mode must be in {lin, log} got '%s' instead" % mode)

    @classmethod
    def scales(cls, domains, mode="lin", xmin=None, xmax=None, resolution=101):
        """
        Generate scales for each domain
        """
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

    def variable_domains(self):
        data = pd.DataFrame(self._xdata)
        return data.describe()

    def variable_scales(self, domains=None, mode="lin", xmin=None, xmax=None, resolution=100):
        """
        Generate Variables Scales
        """
        if domains is None:
            domains = self.variable_domains()
        return self.scales(domains=domains, mode=mode, xmin=xmin, xmax=xmax, resolution=resolution)

    def variable_space(self, domains=None, mode="lin", xmin=None, xmax=None, resolution=10):
        """
        Generate Variable Space
        """
        return np.meshgrid(
            *self.variable_scales(domains=domains, mode=mode, xmin=xmin, xmax=xmax, resolution=resolution)
        )

    def variable_dataset(self, domains=None, mode="lin", xmin=None, xmax=None, resolution=10):
        space = self.variable_space(domains=domains, mode=mode, xmin=xmin, xmax=xmax, resolution=resolution)
        dataset = np.vstack([
            scale.ravel() for scale in space
        ])
        return dataset.T

    def parameter_domains(self, mode="lin", xmin=None, xmax=None, ratio=0.1):
        """
        Generate Parameter Domains
        """

        if self.fitted():
            parameters = self._solution["parameters"]
            if mode == "lin":
                xmin = xmin or list(parameters - 3 * ratio * np.abs(parameters))
                xmax = xmax or list(parameters + 3 * ratio * np.abs(parameters))
            elif mode == "log":
                xmin = xmin or list(parameters * (ratio ** 3))
                xmax = xmax or list(parameters / (ratio ** 3))
            else:
                raise ConfigurationError("Domain mode must be in {lin, log} got '%s' instead" % mode)

        xmin = xmin or 0.
        if not isinstance(xmin, Iterable):
            xmin = [xmin] * self.k

        if len(xmin) != self.k:
            raise ConfigurationError("Domain lower boundaries must have the same dimension as parameter space")

        xmax = xmax or 1.
        if not isinstance(xmax, Iterable):
            xmax = [xmax] * self.k

        if len(xmax) != self.k:
            raise ConfigurationError("Domain upper boundaries must have the same dimension as parameter space")

        return pd.DataFrame([xmin, xmax], index=["min", "max"])

    def parameter_scales(self, domains=None, mode="lin", xmin=None, xmax=None, ratio=0.1, resolution=100):
        """
        Generate Parameter Scales
        """
        if domains is None:
            domains = self.parameter_domains(mode=mode, xmin=xmin, xmax=xmax, ratio=ratio)
        return self.scales(domains=domains, resolution=resolution)

    def parameter_space(self, domains=None, mode="lin", ratio=0.1, xmin=None, xmax=None, resolution=10):
        """
        Generate Parameter Space
        """
        return np.meshgrid(
            *self.parameter_scales(
                domains=domains, mode=mode, ratio=ratio, xmin=xmin, xmax=xmax, resolution=resolution
            )
        )

    def plot_fit(self, title="", resolution=200):
        """
        Plot data and fitted function for each variable
        """

        if self.fitted(error=True):

            scales = self.variable_scales(resolution=resolution)
            for variable_index, scale in enumerate(scales):

                x = self._xdata[:, variable_index]
                xs = scale.reshape(-1, 1)

                fig, axe = plt.subplots()
                axe.plot(
                    x, self._ydata,
                    linestyle="none", marker=".", label=r"Data: $(x_{{{}}},y)$".format(variable_index)
                )
                axe.plot(xs, self.predict(xs), label=r"Fit: $\hat{y} = f(\bar{x},\bar{\beta})$")
                axe.set_title(
                    "Regression Plot: {}\n{}={}, n={:d}, {}={:.3f}, {}={:.3e}".format(
                        title,
                        r"$\bar{\beta}$", np.array2string(self._solution["parameters"], precision=3, separator=', '),
                        self.n, self.score.name, self._score, self.loss.name, self._loss
                    ),
                    fontdict={"fontsize": 11}
                )
                axe.set_xlabel(r"Independent Variable, $x_{{{}}}$".format(variable_index))
                axe.set_ylabel(r"Dependent Variable, $y$")
                axe.legend()
                axe.grid()

                yield axe

    def plot_loss(self, mode="lin", ratio=0.1, xmin=None, xmax=None, title="", levels=None, resolution=200):
        """
        Plot MSE for each parameter pairs
        """

        if self.fitted(error=True):

            scales = self.parameter_scales(
                mode=mode, ratio=ratio, xmin=xmin, xmax=xmax, resolution=resolution
            )

            if self.k > 1:

                for i, j in itertools.combinations(range(self.k), 2):

                    x, y = np.meshgrid(scales[i], scales[j])
                    parameters = list(self._solution["parameters"])
                    parameters[i] = x
                    parameters[j] = y
                    score = self.parametrized_loss()(*parameters)

                    fig, axe = plt.subplots()
                    labels = axe.contour(x, y, np.log10(score), levels or 10, cmap="jet")
                    axe.clabel(labels, labels.levels, inline=True, fontsize=7)

                    axe.axvline(self._solution["parameters"][i], color="black", linestyle="-.")
                    axe.axhline(self._solution["parameters"][j], color="black", linestyle="-.")

                    axe.set_title(
                        "Regression Log-{}: {}\n{}={}, n={:d}, score={:.3e}".format(
                            self.loss.name, title,
                            r"$\bar{\beta}$",
                            np.array2string(self._solution["parameters"], precision=3, separator=', '),
                            self.n, self._loss
                        ),
                        fontdict={"fontsize": 11}
                    )

                    axe.set_xlabel(r"Parameter, $\beta_{{{}}}$".format(i))
                    axe.set_ylabel(r"Parameter, $\beta_{{{}}}$".format(j))
                    axe.grid()

                    axe._pair_indices = (i, j)
                    yield axe

            else:

                scale = scales[0]
                score = self.parametrized_loss()(scale)

                fig, axe = plt.subplots()
                axe.plot(scale, score)
                axe.axvline(self._solution["parameters"][0], color="black", linestyle="-.")

                axe.set_title(
                    "Regression Log-{}: {}\n{}={}, n={:d}, {}={:.3f}, {}={:.3e}".format(
                        self.loss.name, title,
                        r"$\bar{\beta}$",
                        np.array2string(self._solution["parameters"], precision=3, separator=', '),
                        self.n, self.score.name, self._score, self.loss.name, self._loss
                    ),
                    fontdict={"fontsize": 11}
                )

                axe.set_xlabel(r"Parameter, $\beta_0$")
                axe.set_ylabel(r"Score, $s$")
                axe.grid()

                axe._pair_indices = (0, 0)
                yield axe

