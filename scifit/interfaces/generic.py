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
    This class is an interface exposing clean way to fit a defined model to experimental data
    and commodities to analyse regressed parameters and loss function behaviour wrt parameter space.

    This class essentially wraps :py:mod:`scipy` classical optimization procedures to make it easy to use.
    Additionally, the interface is compliant with the
    `SciKit Learn interface <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.base>`_
    in terms of methods and signatures.

    Main goal of :class:`FitSolverInterface` is to regress :math:`k` parameters :math:`(\\beta)` of the model
    to :math:`n` experimental points :math:`(\\textbf{x})` with :math:`m` features (also called variables)
    and :math:`1` target :math:`(y)` by minimizing a loss function :math:`L(\\textbf{x},y,\\beta)`
    which is either the Residual Sum of Squares :math:`(RSS)` if no target uncertainties are given:

    .. math::

       \\arg \\min_{\\beta} RSS = \\sum\\limits_{i=1}^{n}\\left(y_i - \\hat{y}_i\\right)^2 = \\sum\\limits_{i=1}^{n}e_i^2

    Where:

    .. math::

       \\hat{y} = f\\left(x_1, \\dots, x_m, \\beta_0, \\dots, \\beta_k\\right)

    Or more specifically the Chi Square Score :math:`\\chi^2` if target uncertainties are given :math:`(\\sigma_i)`:

    .. math::

       \\arg \\min_{\\beta} \\chi^2 = \\sum\\limits_{i=1}^{n}\\left(\\frac{y_i - \\hat{y}_i}{\\sigma_i}\\right)^2

    Practically, when uncertainties are omitted they are assumed to be unitary, leading to :math:`RSS = \\chi^2`.

    To create a new solver for a specific model, it suffices to subclass the :class:`FitSolverInterface` and
    implement the static method :meth:`scifit.interfaces.generic.FitSolverInterface.model` such in the follwing
    example:

    .. code-block:: python

        from scifit.interfaces.generic import FitSolverInterface

        class LinearFitSolver(FitSolverInterface):
            @staticmethod
            def model(x, a, b):
                return a * x[:, 0] + b

    Which defines a simple linear regression that can be fitted to experimental data.
    """

    def __init__(self, **kwargs):
        """
        Create a new instance of :class:`FitSolverInterface` and store parameters to pass them to :py:mod:`scipy`
        afterwards when fitting experimental data.

        :param kwargs: Dictionary of parameters to pass to :meth:`scipy.optimize.curve_fit`
        """
        self._configuration = kwargs

    def configuration(self, **kwargs):
        """
        Return stored configuration updated by parameters passed to the method.

        :param kwargs: extra parameters to update initially stored configuration
        :return: Dictionary of parameters
        """
        return self._configuration | kwargs

    def store(self, xdata, ydata, sigma=None):
        """
        Validate and store features (variables), target and target uncertainties.

        :param xdata: Experimental features (variables) as a :code:`(n,m)` matrix
        :param ydata: Experimental target as a :code:`(n,)` matrix
        :param sigma: Experimental target uncertainties as :code:`(n,)` matrix
        :raise: Exception :class:`scifit.errors.base.InputDataError` if validation fails
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
    def model(x, *parameters):
        """
        This static method defines the model function to fit to experimental data in order to regress parameters.
        This method must be overridden after the class has been inherited, its signature must be exactly:

        :param x: Features (variables) as :code:`(n,m)` matrix
        :param parameters: Sequence of :code:`k` parameters with explicit names (don't use unpacking when implementing)
        :return: Model function evaluated for the given features and parameters as :code:`(n,)` matrix
        :raise: Exception :class:`scifit.errors.base.MissingModel`
        """
        raise MissingModel("Model must be defined before regression")

    @property
    def observation_space_size(self):
        """
        Number of experimental data (points, observations)
        """
        return self._xdata.shape[0]

    @property
    def n(self):
        """
        Number of experimental data (points, observations)
        """
        return self.observation_space_size

    @property
    def feature_space_size(self):
        """
        Number of features (variables) inferred from experimental data shape
        """
        return self._xdata.shape[1]

    @property
    def m(self):
        """
        Number of features (variables) inferred from experimental data shape
        """
        return self.feature_space_size

    @property
    def signature(self):
        """
        Signature of model function
        """
        return inspect.signature(self.model)

    @property
    def parameter_space_size(self):
        """
        Number of model parameters inferred from model function signature
        """
        return len(self.signature.parameters) - 1

    @property
    def k(self):
        """
        Number of model parameters inferred from model function signature
        """
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
        Generate synthetic target for the model using features, parameters and eventually add noise to it.

        Noise can be added in three different modes (:code:`scale_mode`):

        - :code:`abs` (default) add absolute noise scaled by :code:`sigma`
        - :code:`auto` add absolute noise scaled by :code:`sigma` and halved target extent
        - :code:`rel` add relative noise scaled by :code:`sigma` and target absolute value in each point

        Pseudo Random Generator (PRG) is defined by the object :code:`generator` which must be a PRG
        coming from :py:mod:`numpy.random` by default it is a Standard Normal distribution :class:`numpy.random.normal`.

        :param x: Features (variables) as :code:`(n,m)` matrix
        :param parameters: Sequence of :code:`k` parameters with explicit names
        :param sigma: Shape (scale) parameter for noise if any (default :code:`None`)
        :param precision: Tiny :code:`float` added in noise generation to enforce numerical stability
        :param scale_mode: Type of noise added to target
        :param generator: Pseudo Random Generator used to create noise on target
        :param seed: Seed used by the PRG to sample noise
        :param full_output: Switch to add extra computation to the function output
        :param kwargs: Extra parameters passed to the PRG function
        :return: Target as a :code:`(n,)` matrix when :code:`full_output=False` or a dictionary of objects
                 containing at least synthetic features and details about noise applied on it when :code:`full_output=True`
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
                sigma *= 1.0
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
        Solve the fitting problem by finding a set of parameters minimizing the loss function wrt features, target and sigma.
        Return structured solution and update solver object in order to expose analysis convenience (fit, loss).

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param ydata: Target as :code:`(n,)` matrix
        :param sigma: Uncertainty on target as :code:`(n,)` matrix or scalar or :code:`None`
        :param kwargs: Extra parameters to pass to :code:`scipy.optimize.curve_fit`
        :return: Dictionary of objects with details about the regression including regressed parameters and final covariance
        """
        solution = optimize.curve_fit(
            self.model,
            xdata,
            ydata,
            sigma=sigma,
            absolute_sigma=True,
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
        Boolean switch to indicate if the solver object has stored input data successfully

        :param error: raise error instead of returning
        :return: Stored status as boolean
        :raise: Exception :class:`scifit.errors.base.NotStoredError` if check fails
        """
        is_stored = hasattr(self, "_xdata") and hasattr(self, "_ydata")
        if not (is_stored) and error:
            NotStoredError("Input data must be stored prior to this operation")
        return is_stored

    def fitted(self, error=False):
        """
        Boolean switch to indicate if the fitting procedure has been executed successfully.
        It does not tell anything about the quality and convergence of the actual solution.

        :param error: raise error instead of returning
        :return: Fitted status as boolean
        :raise: Exception :class:`scifit.errors.base.NotFittedError` if check fails
        """
        is_fitted = hasattr(self, "_solution")
        if not (self.stored(error=error)) or not (is_fitted) and error:
            NotFittedError("Model must be fitted prior to this operation")
        return is_fitted

    def solved(self, error=False):
        """
        Boolean switch to indicate if the fitting procedure has been executed successfully
        and has converged to a potential solution.

        :param error: raise error instead of returning
        :return: Fitted status as boolean
        :raise: Exception :class:`scifit.errors.base.NotSolvedError` if check fails
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
        """
        Predict target wrt features (variables) and parameters.
        If parameters are not provided uses regressed parameters (problem needs to be solved first).

        :param x: Features (variables) as :code:`(n,m)` matrix
        :param parameters: Sequence of :code:`k` parameters
        :return: Predicted target as a :code:`(n,)` matrix
        """
        if parameters is not None or self.fitted(error=True):
            return self.model(xdata, *(parameters or self._solution["parameters"]))

    @property
    def degree_of_freedom(self):
        """
        Return the degree of freedom :math:`\\nu` of the actual fitting problem as defined for a
        Chi Square Goodness if Fit Test:

        .. math::

            \\nu = n - k

        Where:

        - :math:`n` is the number of observations
        - :math:`k` is the number of model parameters

        :return: Degree of freedom :math:`\\nu` (strictly positive natural number)
        """
        dof = self.n - self.k
        if dof < 1:
            raise ConfigurationError("Degree of freedom must be greater than zero")
        return dof

    @property
    def dof(self):
        """
        :return: Degree of freedom :math:`\\nu` (strictly positive natural number)
        """
        return self.degree_of_freedom

    def sigma_weight(self, sigma=None):
        """
        Compute uncertainties associated weights as follows:

        .. math::

            w_i = \\frac{1}{\\sigma^2_i}

        :param sigma: Uncertainty on target as :code:`(n,)` matrix or scalar or :code:`None`
        :return: Weights as :code:`(n,)` matrix computed from uncertainties
        """
        if sigma is None:
            sigma = 1.0
        return 1.0/np.power(sigma, 2)

    def WRSS(self, xdata, ydata, wdata=None, parameters=None):
        """
        Compute Weighted Residual Sum of Square (WRSS) which is equivalent to the loss function for this solver.

        .. math::

            WRSS = \\sum\\limits_{i=1}^{n}w_i\cdot \\left(y_i - \\hat{y}_i\\right)^2 = \\sum\\limits_{i=1}^{n}w_i \cdot e_i^2

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param ydata: Target as :code:`(n,)` matrix
        :param wdata: Weights as :code:`(n,)` matrix or scalar or :code:`None`
        :param parameters: Sequence of :code:`k` parameters
        :return: Weighted Residual Sum of Square for given features, target, parameters and target uncertainties
        """
        if wdata is None:
            wdata = 1.0

        return np.sum(
            wdata * np.power(
                (ydata - self.predict(xdata, parameters=parameters)), 2
            )
        )

    def RSS(self, xdata, ydata, parameters=None):
        """
        Compute Residual Sum of Square (RSS) which is equivalent to the loss function for this solver.

        .. math::

            RSS = \\sum\\limits_{i=1}^{n}\\left(y_i - \\hat{y}_i\\right)^2 = \\sum\\limits_{i=1}^{n}e_i^2

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param ydata: Target as :code:`(n,)` matrix
        :param parameters: Sequence of :code:`k` parameters
        :return: Residual Sum of Square for given features, target, parameters and target uncertainties
        """
        return self.WRSS(xdata, ydata, wdata=1.0, parameters=parameters)

    def chi_square(self, xdata, ydata, sigma=None, parameters=None):
        """
        Compute Chi Square Statistic :math:`\\chi^2` reduced to Residual Sum of Squares (RSS) if no sigma provided.

        .. math::

            \\chi^2 = \\sum\\limits_{i=1}^{n}\\left(\\frac{y_i - \\hat{y}_i}{\\sigma_i}\\right)^2

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param ydata: Target as :code:`(n,)` matrix
        :param sigma: Uncertainty on target as :code:`(n,)` matrix or scalar or :code:`None`
        :param parameters: Sequence of :code:`k` parameters
        :return: Chi Square statistic :math:`\\chi^2` for given features, target, parameters and target uncertainties
        """
        return self.WRSS(xdata, ydata, wdata=self.sigma_weight(sigma), parameters=parameters)

    def loss(self, xdata, ydata, sigma=None, parameters=None):
        """
        Compute loss function :math:`L(\\mathbf{x}, y, \\beta)` as Chi Square statistic :math:`\\chi^2`:

        .. math::

            L(\\mathbf{x}, y, \\beta) = \\chi^2 = \\sum\\limits_{i=1}^{n}\\left(\\frac{y_i - \\hat{y}_i}{\\sigma_i}\\right)^2

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param ydata: Target as :code:`(n,)` matrix
        :param sigma: Uncertainty on target as :code:`(n,)` matrix or scalar or :code:`None`
        :param parameters: Sequence of :code:`k` parameters
        :return: Chi Square statistic :math:`\\chi^2` for given features, target, parameters and target uncertainties
        """
        return self.chi_square(xdata, ydata, sigma=sigma, parameters=parameters)

    loss.name = "$\chi^2$"

    def weighted_mean(self, ydata, wdata=None):
        """
        Compute weighted mean as follows:

        .. math::

            \\bar{y}_w = \\frac{ \\sum\\limits_{i=1}^n w_i \cdot y_i}{\\sum\\limits_{i=1}^n w_i}

        :param ydata: Variable as :code:`(n,)` matrix
        :param wdata: Weights as :code:`(n,)` matrix or scalar or :code:`None`
        :return:
        """
        return np.sum(ydata * wdata)/np.sum(wdata)

    def TSS(self, ydata):
        """
        Compute Total Sum of Square (TSS) as follows:

        .. math::

            TSS = \\sum\\limits_{i=1}^{n}\\left(y_i - \\bar{y}\\right)^2

        :param ydata: Variable as :code:`(n,)` matrix
        :return: Total Sum of Square for the given data
        """
        return np.sum(np.power(ydata - ydata.mean(), 2))

    def score(self, xdata, ydata, sigma=None, parameters=None):
        """
        Compute Coefficient of Determination R2
        """
        return 1.0 - self.RSS(xdata, ydata) / self.TSS(ydata)

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
        terms = np.power((ydata - yhat) / sigma, 2)
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
        self._loss = self.loss(self._xdata, self._ydata, sigma=self._sigma)
        self._score = self.score(self._xdata, self._ydata, sigma=self._sigma)
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

    def get_latex_parameters(self, show_sigma=True, precision=2, mode="f"):
        if self.fitted(error=True):
            terms = []
            for i, parameter in enumerate(self._solution["parameters"]):
                term = (r"$\beta_{{{:d}}}=${:.%d%s}" % (precision, mode)).format(
                    i, parameter
                )
                if show_sigma:
                    term += (r" $\pm$ {:.%d%s}" % (precision, mode)).format(
                        np.sqrt(self._solution["covariance"][i][i])
                    )
                # if i % 4 == 0:
                #     term += "\n"
                terms.append(term)
            return ", ".join(terms)

    def get_title(self):
        if self.fitted(error=True):
            full_title = "n={:d}, {}={:.3f}, {}={:.3f}".format(
                self.n,
                self.score.name,
                self._score,
                self.loss.name,
                self._loss,
            )

            if hasattr(self, "_gof"):
                full_title += r", $P(\chi^2_{{{dof:d}}} > {normalized:.3f}) = {pvalue:.4f}$".format(
                    **self._gof
                )

            full_title += "\n" + self.get_latex_parameters()

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

                    axe.set_title(full_title, fontdict={"fontsize": 10})
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

                axe.set_title(full_title, fontdict={"fontsize": 10})
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

                    axe.set_title(full_title, fontdict={"fontsize": 10})
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

                axe.set_title(full_title, fontdict={"fontsize": 10})

                axe.set_xlabel(r"Parameter, $\beta_0$")
                axe.set_ylabel(r"Loss, $L(\beta_0)$")
                axe.grid()

                fig.subplots_adjust(top=0.8, left=0.2)

                axe._pair_indices = (0, 0)
                yield axe
