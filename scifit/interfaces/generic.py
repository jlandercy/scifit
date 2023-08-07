"""
Module :py:mod:`scifit.interfaces.generic` defines the class :class:`GenericInterface`
on which any other interfaces must inherit from. This class exposes generic abstract methods
all interfaces must implement.
"""

import inspect
import itertools
from collections.abc import Iterable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

from scifit.errors.base import *


class FitSolverInterface:

    """
    Generic Interface (Abstract Base Class) for all object of the package.
    This class must be subclassed by any other interfaces.

    n experimental points
    m features
    k parameters
    1 target

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

    def store(self, xdata, ydata, sigma=None):
        """
        Validate and store experimental data
        """

        xdata = np.array(xdata)
        ydata = np.array(ydata)

        if xdata.ndim != 2:
            raise InputDataError("Features must be a two dimensional array")

        if xdata.shape[0] != ydata.shape[0]:
            raise InputDataError(
                "Incompatible shapes between x %s and y %s" % (xdata.shape, ydata.shape)
            )

        if not (
            np.issubdtype(xdata.dtype, np.number)
            and np.issubdtype(ydata.dtype, np.number)
        ):
            raise InputDataError("Input values must be numeric")

        if np.any(np.isnan(xdata)) or np.any(np.isnan(ydata)):
            raise InputDataError("Input values cannot contain missing data (NaN)")

        if isinstance(sigma, Iterable):
            sigma = np.array(sigma)
            if sigma.shape != ydata.shape:
                raise InputDataError("Sigma as array must have the same shape as ydata")
            if not np.all(sigma > 0.0):
                raise InputDataError("All sigma must be strictly positive")
        elif isinstance(sigma, np.number):
            if sigma <= 0.0:
                raise InputDataError("Sigma must be strictly positive")
        elif sigma is None:
            pass
        else:
            raise InputDataError("Sigma must be a number or an array of number")

        self._xdata = xdata
        self._ydata = ydata
        self._sigma = sigma

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
    def feature_space_size(self):
        return self._xdata.shape[1]

    @property
    def m(self):
        return self.feature_space_size

    @property
    def signature(self):
        return inspect.signature(self.model)

    @property
    def parameter_space_size(self):
        return len(self.signature.parameters) - 1

    @property
    def k(self):
        return self.parameter_space_size

    def target_dataset(
        self,
        xdata,
        *parameters,
        sigma=None,
        precision=1e-9,
        scale_mode="abs",
        generator=np.random.normal,
        seed=None,
        full_output=False,
        **kwargs
    ):
        """
        Generate synthetic dataset with additional noise
        """

        if seed is not None:
            np.random.seed(seed)

        yref = self.model(xdata, *parameters)

        if sigma is not None:
            if isinstance(sigma, Iterable):
                sigma = np.array(sigma)
            else:
                sigma = np.full(yref.shape, sigma)

            if scale_mode == "abs":
                sigma *= 1
            elif scale_mode == "auto":
                sigma *= (yref.max() - yref.min()) / 2.0 + precision
            elif scale_mode == "rel":
                sigma *= np.abs(yref) + precision
            else:
                raise ConfigurationError(
                    "Scale must be in {'abs', 'rel', 'auto'}, got '%s' instead"
                    % scale_mode
                )

            ynoise = sigma * generator(size=yref.shape[0], **kwargs)

        else:
            ynoise = np.full(yref.shape, 0.0)

        ydata = yref + ynoise

        if full_output:
            return {
                "xdata": xdata,
                "parameters": np.array(parameters),
                "yref": yref,
                "sigmas": sigma,
                "ynoise": ynoise,
                "ydata": ydata,
            }

        return ydata

    def solve(self, xdata, ydata, sigma=None, **kwargs):
        """
        Solve fitting problem and return structured solution
        """
        solution = optimize.curve_fit(
            self.model,
            xdata,
            ydata,
            sigma=sigma,
            full_output=True,
            check_finite=True,
            nan_policy="raise",
            **self.configuration(**kwargs)
        )
        return {
            "parameters": solution[0],
            "covariance": solution[1],
            "info": solution[2],
            "message": solution[3],
            "status": solution[4],
        }

    def stored(self, error=False):
        """
        Stored means the solver object has stored input data successfully
        """
        is_stored = hasattr(self, "_xdata") and hasattr(self, "_ydata")
        if not (is_stored) and error:
            NotFittedError("Input data must be stored prior to this operation")
        return is_stored

    def fitted(self, error=False):
        """
        Fitted means the fitting procedure has been executed successfully
        """
        is_fitted = hasattr(self, "_solution")
        if not (self.stored(error=error)) or not (is_fitted) and error:
            NotFittedError("Model must be fitted prior to this operation")
        return is_fitted

    def solved(self, error=False):
        """
        Solved means the fitting procedure has been executed successfully and has converged to a potential solution
        """
        has_converged = self.fitted(error=error) and self._solution["status"] in {
            1,
            2,
            3,
            4,
        }
        if not (has_converged) and error:
            NotSolvedError(
                "Fitting procedure has not converged ({status:}): {message:}".format(
                    **self._solution
                )
            )
        return has_converged

    def predict(self, xdata, parameters=None):
        if parameters is not None or self.fitted(error=True):
            return self.model(xdata, *(parameters or self._solution["parameters"]))

    @property
    def degree_of_freedom(self):
        dof = self.n - self.k
        if dof < 1:
            raise ConfigurationError("Degree of freedom must be greater than zero")
        return dof

    @property
    def dof(self):
        return self.degree_of_freedom

    def loss(self, xdata, ydata, sigma=None, parameters=None):
        """
        Compute Mean Squared Error
        """
        if sigma is None:
            sigma = 1.0
        return (
            np.sum(
                np.power(
                    (ydata - self.predict(xdata, parameters=parameters)) / sigma, 2
                )
            )
            / self.dof
        )

    loss.name = "$\chi^2$"

    def score(self, xdata, ydata, parameters=None):
        """
        Compute Coefficient of Determination R2
        """
        RSS = np.sum(np.power(ydata - self.predict(xdata, parameters=parameters), 2))
        TSS = np.sum(np.power(ydata - self._ydata.mean(), 2))
        return 1 - RSS / TSS

    score.name = "$R^2$"

    def goodness_of_fit(
        self, xdata, ydata, sigma=None, parameters=None, full_output=False
    ):
        """
        Compute Chi Square for Goodness of Fit
        """
        yhat = self.predict(xdata, parameters=parameters)
        if sigma is None:
            sigma = 1.0
        terms = np.power(yhat - ydata, 2) / np.power(sigma, 2)
        statistic = np.sum(terms)
        normalized = statistic / self.n
        law = stats.chi2(df=self.dof)
        result = {
            "n": self.n,
            "k": self.k,
            "dof": self.dof,
            "statistic": statistic,
            "normalized": normalized,
            "pvalue": law.sf(statistic),
            "quantile": law.cdf(statistic),
            "P95": law.ppf(0.95),
            "P99": law.ppf(0.99),
            "P999": law.ppf(0.999),
            "law": law,
        }
        if full_output:
            result.update(
                {
                    "xdata": xdata,
                    "ydata": ydata,
                    "sigma": sigma,
                    "yhat": yhat,
                    "terms": terms,
                }
            )
        return result

    def parametrized_loss(self, sigma=None):
        """
        Vectorized loss wrt to parameter space
        """

        @np.vectorize
        def wrapped(*parameters):
            return self.loss(
                self._xdata, self._ydata, sigma=sigma, parameters=parameters
            )

        return wrapped

    def fit(self, xdata, ydata, sigma=None, **kwargs):
        """
        Solve fitting problem and store data and results
        """
        self.store(xdata, ydata, sigma=sigma)
        self._solution = self.solve(
            self._xdata, self._ydata, sigma=self._sigma, **kwargs
        )
        self._yhat = self.predict(self._xdata)
        self._loss = self.loss(self._xdata, self._ydata)
        self._score = self.score(self._xdata, self._ydata)
        self._gof = self.goodness_of_fit(self._xdata, self._ydata, sigma=self._sigma)
        return self._solution

    @staticmethod
    def scale(mode="lin", xmin=0.0, xmax=1.0, resolution=100, base=10):
        """
        Generate 1-dimensional scale
        """
        if mode == "lin":
            return np.linspace(xmin, xmax, resolution)
        elif mode == "log":
            return np.logspace(xmin, xmax, resolution, base=base)
        else:
            raise ConfigurationError(
                "Domain mode must be in {lin, log} got '%s' instead" % mode
            )

    @classmethod
    def scales(
        cls,
        domains=None,
        mode="lin",
        xmin=None,
        xmax=None,
        dimension=None,
        resolution=100,
    ):
        """
        Generate scales for each domain or synthetic scales if no domains defined
        """
        if (domains is None) and ((xmin is None) or (xmax is None)):
            ConfigurationError(
                "Scales requires at least domains or xmin and xmax to be defined"
            )

        if domains is None:
            xmin, xmax = [xmin or 0.0] * dimension, [xmax or 1.0] * dimension
        else:
            xmin = domains.loc["min", :]
            xmax = domains.loc["max", :]

        scales = [
            cls.scale(mode=mode, xmin=xmin[i], xmax=xmax[i], resolution=resolution)
            for i in range(dimension or domains.shape[1])
        ]
        return scales

    def feature_domains(self):
        data = pd.DataFrame(self._xdata)
        return data.describe()

    def feature_scales(
        self,
        domains=None,
        mode="lin",
        xmin=None,
        xmax=None,
        dimension=None,
        resolution=100,
    ):
        """
        Generate Features Scales
        """
        if (dimension is None) and (domains is None):
            domains = self.feature_domains()
        return self.scales(
            domains=domains,
            mode=mode,
            xmin=xmin,
            xmax=xmax,
            dimension=dimension,
            resolution=resolution,
        )

    def feature_space(
        self,
        domains=None,
        mode="lin",
        xmin=None,
        xmax=None,
        dimension=None,
        resolution=10,
    ):
        """
        Generate Feature Space
        """
        return np.meshgrid(
            *self.feature_scales(
                domains=domains,
                mode=mode,
                xmin=xmin,
                xmax=xmax,
                dimension=dimension,
                resolution=resolution,
            )
        )

    def feature_dataset(
        self,
        domains=None,
        mode="lin",
        xmin=None,
        xmax=None,
        dimension=None,
        resolution=10,
    ):
        space = self.feature_space(
            domains=domains,
            mode=mode,
            xmin=xmin,
            xmax=xmax,
            dimension=dimension,
            resolution=resolution,
        )
        dataset = np.vstack([scale.ravel() for scale in space])
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
                xmin = xmin or list(parameters * (ratio**3))
                xmax = xmax or list(parameters / (ratio**3))
            else:
                raise ConfigurationError(
                    "Domain mode must be in {lin, log} got '%s' instead" % mode
                )

        xmin = xmin or 0.0
        if not isinstance(xmin, Iterable):
            xmin = [xmin] * self.k

        if len(xmin) != self.k:
            raise ConfigurationError(
                "Domain lower boundaries must have the same dimension as parameter space"
            )

        xmax = xmax or 1.0
        if not isinstance(xmax, Iterable):
            xmax = [xmax] * self.k

        if len(xmax) != self.k:
            raise ConfigurationError(
                "Domain upper boundaries must have the same dimension as parameter space"
            )

        return pd.DataFrame([xmin, xmax], index=["min", "max"])

    def parameter_scales(
        self, domains=None, mode="lin", xmin=None, xmax=None, ratio=0.1, resolution=100
    ):
        """
        Generate Parameter Scales
        """
        if domains is None:
            domains = self.parameter_domains(
                mode=mode, xmin=xmin, xmax=xmax, ratio=ratio
            )
        return self.scales(domains=domains, resolution=resolution)

    def parameter_space(
        self, domains=None, mode="lin", ratio=0.1, xmin=None, xmax=None, resolution=10
    ):
        """
        Generate Parameter Space
        """
        return np.meshgrid(
            *self.parameter_scales(
                domains=domains,
                mode=mode,
                ratio=ratio,
                xmin=xmin,
                xmax=xmax,
                resolution=resolution,
            )
        )

    def get_title(self):
        if self.fitted(error=True):
            full_title = "{}={}, n={:d}, {}={:.3f}, {}={:.3e}".format(
                r"$\bar{\beta}$",
                np.array2string(
                    self._solution["parameters"], precision=2, separator=", "
                ),
                self.n,
                self.score.name,
                self._score,
                self.loss.name,
                self._loss,
            )

            if hasattr(self, "_gof"):
                full_title += (
                    "\n"
                    + r"$\chi^2_{{{dof:d}}} = {normalized:.3f}, p = {pvalue:.4f}$".format(
                        **self._gof
                    )
                )

        return full_title

    def plot_fit(
        self,
        title="",
        errors=False,
        squared_errors=False,
        aspect="auto",
        resolution=200,
    ):
        """
        Plot data and fitted function for each feature
        """

        if self.fitted(error=True):
            full_title = "Fit Plot: {}\n{}".format(title, self.get_title())
            if self.m == 1:
                scales = self.feature_scales(resolution=resolution)
                for feature_index, scale in enumerate(scales):
                    xdata = self._xdata[:, feature_index]
                    error = self._ydata - self._yhat
                    xscale = scale.reshape(-1, 1)

                    fig, axe = plt.subplots()

                    if self._sigma is None:
                        axe.plot(
                            xdata,
                            self._ydata,
                            linestyle="none",
                            marker=".",
                            label=r"Data: $(x_{{{}}},y)$".format(feature_index),
                        )
                    else:
                        axe.errorbar(
                            xdata,
                            self._ydata,
                            yerr=self._sigma,
                            lolims=False,
                            uplims=False,
                            linestyle="none",
                            marker=".",
                            label=r"Data: $(x_{{{}}},y)$".format(feature_index),
                        )

                    axe.plot(
                        xscale,
                        self.predict(xscale),
                        label=r"Fit: $\hat{y} = f(\bar{x},\bar{\beta})$",
                    )

                    if errors:
                        for xs, y, e in zip(xdata, self._ydata, error):
                            axe.plot([xs, xs], [y, y - e], color="blue", linewidth=0.25)

                    if squared_errors:
                        for xs, y, e in zip(xdata, self._ydata, error):
                            square = patches.Rectangle(
                                (xs, y),
                                -e,
                                -e,
                                linewidth=0.0,
                                edgecolor="black",
                                facecolor="lightblue",
                                alpha=0.5,
                            )
                            axe.add_patch(square)

                    axe.set_title(full_title, fontdict={"fontsize": 11})
                    axe.set_xlabel(r"Feature, $x_{{{}}}$".format(feature_index))
                    axe.set_ylabel(r"Target, $y$")
                    axe.set_aspect(aspect)
                    axe.legend()
                    axe.grid()

                    fig.subplots_adjust(top=0.8, left=0.2)
                    # fig.tight_layout()

                    yield axe

            elif self.m == 2:
                fig = plt.figure()
                axe = fig.add_subplot(projection="3d")

                axe.scatter(*self._xdata.T, self._ydata)

                domains = self.feature_domains()
                X0, X1 = self.feature_space(domains=domains, resolution=200)
                xs = self.feature_dataset(domains=domains, resolution=200)
                ys = self.predict(xs)
                Ys = ys.reshape(X0.shape)

                axe.plot_surface(
                    X0, X1, Ys, cmap="jet", linewidth=0.0, alpha=0.5, antialiased=True
                )

                if errors:
                    for x0, x1, y, e in zip(
                        *self._xdata.T, self._ydata, self._ydata - self._yhat
                    ):
                        axe.plot(
                            [x0, x0], [x1, x1], [y, y - e], color="blue", linewidth=0.5
                        )

                axe.set_title(full_title, fontdict={"fontsize": 11})
                axe.set_xlabel(r"Feature, $x_0$")
                axe.set_ylabel(r"Feature, $x_1$")
                axe.set_zlabel(r"Target, $y$")
                axe.grid()

                fig.subplots_adjust(top=0.8, left=0.2)

                yield axe

            else:
                pass

    def plot_loss(
        self,
        mode="lin",
        ratio=0.1,
        xmin=None,
        xmax=None,
        title="",
        levels=None,
        resolution=200,
    ):
        """
        Plot loss function for each parameter pairs
        """

        if self.fitted(error=True):
            full_title = "Fit Loss Plot: {}\n{}".format(title, self.get_title())

            scales = self.parameter_scales(
                mode=mode, ratio=ratio, xmin=xmin, xmax=xmax, resolution=resolution
            )

            if self.k > 1:
                for i, j in itertools.combinations(range(self.k), 2):
                    x, y = np.meshgrid(scales[i], scales[j])
                    parameters = list(self._solution["parameters"])
                    parameters[i] = x
                    parameters[j] = y
                    score = self.parametrized_loss(sigma=self._sigma)(*parameters)

                    fig, axe = plt.subplots()
                    labels = axe.contour(x, y, score, levels or 10, cmap="jet")
                    axe.clabel(labels, labels.levels, inline=True, fontsize=7)

                    axe.axvline(
                        self._solution["parameters"][i], color="black", linestyle="-."
                    )
                    axe.axhline(
                        self._solution["parameters"][j], color="black", linestyle="-."
                    )

                    axe.set_title(full_title, fontdict={"fontsize": 11})
                    axe.set_xlabel(r"Parameter, $\beta_{{{}}}$".format(i))
                    axe.set_ylabel(r"Parameter, $\beta_{{{}}}$".format(j))
                    axe.grid()

                    fig.subplots_adjust(top=0.8, left=0.2)

                    axe._pair_indices = (i, j)
                    yield axe

            else:
                scale = scales[0]
                score = self.parametrized_loss(sigma=self._sigma)(scale)

                fig, axe = plt.subplots()
                axe.plot(scale, score)
                axe.axvline(
                    self._solution["parameters"][0], color="black", linestyle="-."
                )

                axe.set_title(full_title, fontdict={"fontsize": 11})

                axe.set_xlabel(r"Parameter, $\beta_0$")
                axe.set_ylabel(r"Loss, $L(\beta_0)$")
                axe.grid()

                fig.subplots_adjust(top=0.8, left=0.2)

                axe._pair_indices = (0, 0)
                yield axe
