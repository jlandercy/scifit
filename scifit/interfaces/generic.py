import inspect
import itertools
import logging
import numbers
from collections.abc import Iterable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

from scifit.errors.base import *

logger = logging.getLogger(__name__)


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
        elif isinstance(sigma, numbers.Number):
            if sigma <= 0.0:
                raise InputDataError("Sigma must be strictly positive")
        elif sigma is None:
            pass
        else:
            raise InputDataError("Sigma must be a number or an array of number")

        self._xdata = xdata
        self._ydata = ydata
        self._sigma = sigma

        logger.info(
            "Stored data into solver: X%s, y%s, s=%s"
            % (xdata.shape, ydata.shape, sigma is not None)
        )

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

    def generate_noise(
        self,
        yref,
        sigma=None,
        precision=1e-9,
        scale_mode="abs",
        generator=np.random.normal,
        seed=None,
        full_output=True,
        **kwargs,
    ):
        if seed is not None:
            np.random.seed(seed)

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

            noise = sigma * generator(size=yref.shape[0], **kwargs)

        else:
            noise = np.full(yref.shape, 0.0)

        if full_output:
            return {
                "noise": noise,
                "sigmas": sigma,
            }
        else:
            return noise

    def target_dataset(
        self,
        xdata=None,
        parameters=None,
        dimension=1,
        mode="lin",
        xmin=-1.0,
        xmax=1.0,
        resolution=30,
        sigma=None,
        precision=1e-9,
        scale_mode="abs",
        generator=np.random.normal,
        seed=None,
        full_output=False,
        **kwargs,
    ):
        """
        Generate synthetic target for the model using features, parameters and eventually add noise to it.

        Noise can be added in three different modes (:code:`scale_mode`):

        - :code:`abs` (default) add absolute noise scaled by :code:`sigma`
        - :code:`auto` add absolute noise scaled by :code:`sigma` and halved target extent
        - :code:`rel` add relative noise scaled by :code:`sigma` and target absolute value in each point

        Pseudo Random Generator (PRG) is defined by the object :code:`generator` which must be a PRG
        coming from :py:mod:`numpy.random` by default it is a Standard Normal distribution :class:`numpy.random.normal`.

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param parameters: Sequence of :code:`k` parameters with explicit names
        :param dimension:
        :param mode:
        :param xmin:
        :param xmax:
        :param resolution:
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

        if xdata is None:
            xdata = self.feature_dataset(
                mode=mode,
                xmin=xmin,
                xmax=xmax,
                dimension=dimension,
                resolution=resolution,
            )

        if parameters is None:
            parameters = np.power(np.random.uniform(size=(self.k,)), 2) + 1.0

        yref = self.model(xdata, *parameters)
        noise = self.generate_noise(
            yref,
            sigma=sigma,
            precision=precision,
            scale_mode=scale_mode,
            generator=generator,
            seed=seed,
            full_output=True,
            **kwargs,
        )
        ydata = yref + noise["noise"]

        if full_output:
            return {
                "parameters": np.array(parameters),
                "x": xdata,
                "y": ydata,
                "sy": noise["sigmas"],
                "yref": yref,
                "ynoise": noise["noise"],
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

        # Adapt signature for single number:
        if isinstance(sigma, numbers.Number):
            sigma = np.full(ydata.shape, float(sigma))

        solution = optimize.curve_fit(
            self.model,
            xdata,
            ydata,
            sigma=sigma,
            absolute_sigma=True,
            full_output=True,
            check_finite=True,
            nan_policy="raise",
            **self.configuration(**kwargs),
        )

        return {
            "success": solution[4] in {1, 2, 3, 4},
            "parameters": solution[0],
            "covariance": solution[1],
            "info": solution[2],
            "message": solution[3],
            "status": solution[4],
        }

    @staticmethod
    def minimize_callback(result):
        pass

    def minimize(self, xdata, ydata, sigma=None, p0=None, **kwargs):
        """
        Solve the fitting problem by finding a set of parameters minimizing the loss function wrt features, target and sigma.
        Return structured solution and update solver object in order to expose analysis convenience (fit, loss).

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param ydata: Target as :code:`(n,)` matrix
        :param sigma: Uncertainty on target as :code:`(n,)` matrix or scalar or :code:`None`
        :param kwargs: Extra parameters to pass to :code:`scipy.optimize.curve_fit`
        :return: Dictionary of objects with details about the regression including regressed parameters and final covariance
        """

        # Adapt default parameters as curve_fit
        if p0 is None:
            p0 = np.full((self.k,), 1.0)

        # Adapt signature for single number:
        if isinstance(sigma, numbers.Number):
            sigma = np.full(ydata.shape, sigma)

        # Adapt loss function signature:
        def loss(p):
            return self.parametrized_loss(xdata, ydata, sigma=sigma)(*p)

        def callback(result):
            self._iterations.append(result)
            self.minimize_callback(result)

        self._iterations = []
        solution = optimize.minimize(
            loss, x0=p0, method="L-BFGS-B", jac="3-point", callback=callback, **kwargs
        )
        self._iterations = np.array(self._iterations)

        # # From scipy: https://github.com/scipy/scipy/blob/main/scipy/optimize/_minpack_py.py#L1000C1-L1004C39
        # _, s, VT = np.linalg.svd(solution.jac, full_matrices=False)
        # threshold = np.finfo(float).eps * max(solution.jac.shape) * s[0]
        # s = s[s > threshold]
        # VT = VT[:s.size]
        # covariance = np.dot(VT.T / s**2, VT)

        return {
            "success": solution.success,
            "parameters": solution.x,
            "covariance": None,
            "info": {
                "jac": solution.jac,
                "nit": solution.nit,
                "fun": solution.fun,
                "nfev": solution.nfev,
                "njev": solution.njev,
                "hess_inv": solution.hess_inv,
                "iterations": self._iterations,
            },
            "message": solution.message,
            "status": solution.status,
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
        return 1.0 / np.power(sigma, 2)

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
            wdata * np.power((ydata - self.predict(xdata, parameters=parameters)), 2)
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
        return self.WRSS(
            xdata, ydata, wdata=self.sigma_weight(sigma), parameters=parameters
        )

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

    loss.name = "$\chi^2_r$"

    def weighted_mean(self, ydata, wdata=None):
        """
        Compute weighted mean as follows:

        .. math::

            \\bar{y}_w = \\frac{ \\sum\\limits_{i=1}^n w_i \cdot y_i}{\\sum\\limits_{i=1}^n w_i}

        :param ydata: Variable as :code:`(n,)` matrix
        :param wdata: Weights as :code:`(n,)` matrix or scalar or :code:`None`
        :return:
        """
        return np.sum(ydata * wdata) / np.sum(wdata)

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
        Compute Coefficient of Determination :math:`R^2` as follows:

        .. math::

            R^2 = 1 - \\frac{RSS}{TSS}

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param ydata: Target as :code:`(n,)` matrix
        :param sigma: Uncertainty on target as :code:`(n,)` matrix or scalar or :code:`None`
        :param parameters: Sequence of :code:`k` parameters
        :return: Coefficient of Determination :math:`R^2`
        """
        return 1.0 - self.RSS(xdata, ydata) / self.TSS(ydata)

    score.name = "$R^2$"

    def goodness_of_fit(self, xdata, ydata, sigma=None, parameters=None):
        """
        Compute Chi Square for Goodness of Fit (GoF) as follows:

        - Create a Chi Square distribution :math:`\\chi^2(\\nu=n-k)`
        - Assess critical values :math:`\\chi^2_c` for typical values of :math:`\\alpha = 0.05, 0.01, 0.005, 0.001` for
          left and right sided tests:

        .. math::

            P\\left[\\chi^2 \\leq \\chi^2_c\\right] = \\int\\limits_{-\\infty}^{\\chi^2_c}f(x)\\mathrm{d}x = \\alpha \\Rightarrow \\chi^2_c = \\mathrm{ppf}(\\alpha)

        .. math::

            P\\left[\\chi^2 \\geq \\chi^2_c\\right] = \\int\\limits_{\\chi^2_c}^{+\\infty}f(x)\\mathrm{d}x = \\alpha \\Rightarrow \\chi^2_c = \\mathrm{ppf}(1 - \\alpha)

        - Compute actual Chi Square statistic :math:`\\chi^2_r` and the :math:`p`-value associated:

        .. math::

            p = P\\left[\\chi^2 \\geq \\chi^2_r\\right] = \\int\\limits_{\\chi^2_r}^{+\\infty}f(x)\\mathrm{d}x = 1 - \\mathrm{cdf}(\\chi^2_r) = \\mathrm{sf}(\\chi^2_r)

        - Check for typical values of :math:`\\alpha` if :math:`H_0` must be rejected or not in three modes:
          left, right and both sided.

        Figure below shows a chi square tests for an adjustment:

        .. image:: ../media/figures/GoodnessOfFitPlot.png
          :width: 560
          :alt: Chi Square Goodness of Fit Plot

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param ydata: Target as :code:`(n,)` matrix
        :param sigma: Uncertainty on target as :code:`(n,)` matrix or scalar or :code:`None`
        :param parameters: Sequence of :code:`k` parameters
        :return: Dictionary of objects containing elements to interpret the Chi Square Test for Goodness of Fit
        """
        statistic = self.chi_square(xdata, ydata, sigma=sigma, parameters=parameters)
        normalized = statistic / self.dof
        law = stats.chi2(df=self.dof)
        result = {
            "n": self.n,
            "k": self.k,
            "dof": self.dof,
            "statistic": statistic,
            "normalized": normalized,
            "pvalue": law.sf(statistic),
            "law": law,
            "significance": {
                "left-sided": [],
                "right-sided": [],
                "two-sided": [],
            },
        }
        for alpha in [0.500, 0.100, 0.050, 0.010, 0.005, 0.001]:
            # Left Sided Test:
            chi = law.ppf(alpha)
            result["significance"]["left-sided"].append(
                {"alpha": alpha, "value": chi, "H0": chi <= statistic}
            )
            # Right Sided Test:
            chi = law.ppf(1.0 - alpha)
            result["significance"]["right-sided"].append(
                {"alpha": alpha, "value": chi, "H0": statistic <= chi}
            )
            # Two Sided Test:
            low = law.ppf(alpha / 2.0)
            high = law.ppf(1.0 - alpha / 2.0)
            result["significance"]["two-sided"].append(
                {
                    "alpha": alpha,
                    "low-value": low,
                    "high-value": high,
                    "H0": low <= statistic <= high,
                }
            )
        return result

    def parametrized_loss(self, xdata=None, ydata=None, sigma=None):
        """
        **Wrapper:** Loss function decorated with experimental data and vectorized for parameters.
        This decorator load loss method with features (variables), target and sigma to expose only parameters.
        Decorated function is used by :meth:`FitSolverInterface.plot_loss` in order to sketch the loss landscape.

        .. code-block:: python

            loss = solver.parametrized_loss(X, y, sigma)
            chi2 = loss(*parameters)

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param ydata: Target as :code:`(n,)` matrix
        :param sigma: Uncertainty on target as :code:`(n,)` matrix or scalar or :code:`None`
        :return: Wrapped loss function decorated with experimental data and vectorized for parameters
        """

        if xdata is None:
            xdata = self._xdata
        if ydata is None:
            ydata = self._ydata
        if sigma is None:
            sigma = self._sigma

        @np.vectorize
        def wrapped(*parameters):
            return self.loss(xdata, ydata, sigma=sigma, parameters=parameters)

        return wrapped

    def fit(self, xdata=None, ydata=None, sigma=None, **kwargs):
        """
        Fully solve the fitting problem for the given model and input data.
        This method stores input data and fit results. It assesses loss function over parameter neighborhoods,
        computes score and performs goodness of fit test.

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param ydata: Target as :code:`(n,)` matrix
        :param sigma: Uncertainty on target as :code:`(n,)` matrix or scalar or :code:`None`
        :param kwargs: Dictionary of extra parameters passed to :py:mod`curve_fit`.
        :return: Dictionary of values related to fit solution

        """

        if not self.stored(error=False):
            self.store(xdata, ydata, sigma=sigma)

        self._solution = self.solve(
            self._xdata, self._ydata, sigma=self._sigma, **kwargs
        )
        # self._minimize = self.minimize(
        #     self._xdata, self._ydata, sigma=self._sigma, **kwargs
        # )

        self._yhat = self.predict(self._xdata)
        self._loss = self.loss(self._xdata, self._ydata, sigma=self._sigma)
        self._score = self.score(self._xdata, self._ydata, sigma=self._sigma)
        self._gof = self.goodness_of_fit(self._xdata, self._ydata, sigma=self._sigma)

        return self._solution

    @staticmethod
    def scale(mode="lin", xmin=0.0, xmax=1.0, resolution=100, base=10):
        """
        Generate a 1-dimensional scale (xmin, xmax) of given resolution and mode
        """
        if mode == "lin":
            return np.linspace(xmin, xmax, resolution)
        elif mode == "log":
            xmin = np.log10(xmin) / np.log10(base)
            xmax = np.log10(xmax) / np.log10(base)
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
        Generate scales for each domain or synthetic scales if domains are not defined
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
        """
        Get feature (variable) domains, useful for drawing scales fitting the dataset
        :return:
        """
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
        Generate features scales
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
        Generate feature space
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
        """
        Generate feature dataset, useful to generate variables for a synthetic dataset or to create a grid
        for interpolation over the features space.
        """
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

    def parameter_domains(
        self,
        parameters=None,
        mode="lin",
        xmin=None,
        xmax=None,
        ratio=10.0,
        factor=3.0,
    ):
        """
        Generate parameter domains, useful for drawing scales fitting the parameters space
        """

        if mode not in {"lin", "log"}:
            raise ConfigurationError(
                "Domain mode must be in {lin, log} got '%s' instead" % mode
            )

        if parameters is None and self.fitted(error=True):
            parameters = self._solution["parameters"]

        if parameters is not None:
            if mode == "lin":
                xmin = xmin or list(parameters - factor * ratio * np.abs(parameters))
                xmax = xmax or list(parameters + factor * ratio * np.abs(parameters))

            elif mode == "log":
                xmin = xmin or list(parameters / np.power(ratio, factor))
                xmax = xmax or list(parameters * np.power(ratio, factor))

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
        Generate parameter scales
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
        Generate parameter space
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

    def get_latex_parameters(self, show_sigma=True, precision=3, mode="f"):
        """
        Return parameters in a compact LaTeX fashion, useful for figure title
        """
        if self.fitted(error=True):
            terms = []
            for i, parameter in enumerate(self._solution["parameters"]):
                term = ("{:.%d%s}" % (precision, mode)).format(parameter)
                if show_sigma:
                    term += (r" \pm {:.%d%s}" % (precision, mode)).format(
                        np.sqrt(self._solution["covariance"][i][i])
                    )
                # if i % 4 == 0:
                #     term += "\n"
                terms.append(term)
            return r"$\beta = ({})$".format(", ".join(terms))

    def get_title(self):
        """
        Get detailed figure title (including parameters if fitted)
        """
        if self.fitted(error=True):
            full_title = "n={:d}, {}={:.3f}, {}={:.3f}".format(
                self.n,
                self.score.name,
                self._score,
                self.loss.name,
                self._loss,
            )

            if hasattr(self, "_gof"):
                full_title += r", $P(\chi^2_{{{dof:d}}} \geq {normalized:.3f} \,|\, H_0) = {pvalue:.4f}$".format(
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
        resolution=100,
        mode="lin",
        log_x=False,
        log_y=False,
    ):
        """
        Plot data and fitted function for low dimension problems (:math:`m \leq 2`)

        .. image:: ../media/figures/FitPlot.png
            :width: 560
            :alt: Fit Plot
        """

        if log_x:
            mode = "log"

        if self.fitted(error=True):
            full_title = "Fit Plot: {}\n{}".format(title, self.get_title())
            if self.m == 1:
                scales = self.feature_scales(resolution=resolution, mode=mode)

                xdata = self._xdata[:, 0]
                error = self._ydata - self._yhat
                xscale = scales[0].reshape(-1, 1)

                fig, axe = plt.subplots()

                if self._sigma is None:
                    axe.plot(
                        xdata,
                        self._ydata,
                        linestyle="none",
                        marker=".",
                        label=r"Data: $(x_{1},y)$",
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
                        label=r"Data: $(x_1,y)$",
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
                axe.set_xlabel(r"Feature, $x_1$")
                axe.set_ylabel(r"Target, $y$")
                axe.set_aspect(aspect)
                axe.legend()
                axe.grid()

                if log_x:
                    axe.set_xscale("log")

                if log_y:
                    axe.set_yscale("log")

                fig.subplots_adjust(top=0.8, left=0.2)

                return axe

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
                axe.set_xlabel(r"Feature, $x_1$")
                axe.set_ylabel(r"Feature, $x_2$")
                axe.set_zlabel(r"Target, $y$")
                axe.grid()

                fig.subplots_adjust(top=0.8, left=0.2)

                return axe

            else:
                pass

    def plot_chi_square(self, title="", resolution=100):
        """
        Plot Chi Square Goodness of Fit figure, summarizes all critical thresholds and p-value

        .. image:: ../media/figures/GoodnessOfFitPlot.png
            :width: 560
            :alt: Chi Square Goodness of Fit Plot


        :param title:
        :param resolution:
        :return:
        """
        if self.fitted(error=True):
            full_title = "Fit $\chi^2$ Plot: {}\n{}".format(title, self.get_title())

            law = self._gof["law"]
            statistic = self._gof["statistic"]
            xlin = np.linspace(law.ppf(0.0001), law.ppf(0.9999), resolution)
            xarea = np.linspace(statistic, law.ppf(0.9999), resolution)

            fig, axe = plt.subplots()

            axe.plot(
                xlin,
                law.pdf(xlin),
                label=r"$\chi^2(\nu={:d})$".format(self._gof["dof"]),
            )
            axe.fill_between(
                xarea,
                law.pdf(xarea),
                alpha=0.5,
                label=r"$p$ = {:.4f}".format(self._gof["pvalue"]),
            )
            axe.axvline(
                statistic,
                linestyle="-.",
                color="black",
                label=r"$\chi^2_r = {:.3f}$".format(statistic),
            )

            alphas = [0.050, 0.025, 0.010, 0.005]
            colors = ["orange", "darkorange", "red", "darkred"]

            for alpha, color in zip(reversed(alphas), reversed(colors)):
                chi2 = law.ppf(alpha)
                axe.axvline(
                    chi2,
                    color=color,
                    linestyle="--",
                    label=r"$\chi^2_{{\alpha = {:.1f}\%}} = {:.1f}$".format(
                        alpha * 100.0, chi2
                    ),
                )

            for alpha, color in zip(alphas, colors):
                chi2 = law.ppf(1.0 - alpha)
                axe.axvline(
                    chi2,
                    color=color,
                    label=r"$\chi^2_{{\alpha = {:.1f}\%}} = {:.1f}$".format(
                        alpha * 100.0, chi2
                    ),
                )

            axe.set_title(full_title, fontdict={"fontsize": 10})
            axe.set_xlabel(r"Random Variable, $\chi^2$")
            axe.set_ylabel(r"Density, $f(\chi^2)$")
            axe.legend(bbox_to_anchor=(1, 1), fontsize=8)
            axe.grid()

            fig.subplots_adjust(top=0.8, left=0.15, right=0.75)

            return axe

    def plot_loss_low_dimension(
        self,
        first_index=None,
        second_index=None,
        axe=None,
        mode="lin",
        ratio=0.1,
        xmin=None,
        xmax=None,
        title="",
        levels=None,
        resolution=75,
        surface=False,
        iterations=False,
        add_labels=True,
        add_title=True,
        log_x=False,
        log_y=False,
        log_loss=False,
    ):
        """
        Sketch and plot loss function for low dimensional space (complete for :math:`k \leq 2`) or sub-space.

        .. image:: ../media/figures/FitLossPlotLowDim.png
            :width: 560
            :alt: Low Dimensionality Fit Loss Plot

        See :meth:`FitSolverInterface.plot_loss` for full loss landscape.
        """

        if self.fitted(error=True):
            if axe is None:
                if surface:
                    fig, axe = plt.subplots(subplot_kw={"projection": "3d"})
                else:
                    fig, axe = plt.subplots()
            fig = axe.figure

            full_title = "Fit Loss Plot: {}\n{}".format(title, self.get_title())

            scales = self.parameter_scales(
                mode=mode, ratio=ratio, xmin=xmin, xmax=xmax, resolution=resolution
            )

            if self.k == 1 or (first_index is not None and second_index is None):
                first_index = first_index or 0
                p0 = self._solution["parameters"]
                parameters = list(p0)
                parameters[first_index] = scales[first_index]
                loss = self.parametrized_loss(sigma=self._sigma)

                if log_loss:
                    loss = np.log10(loss)

                fig, axe = plt.subplots()

                axe.plot(scales[first_index], loss(*parameters))
                axe.axvline(
                    self._solution["parameters"][first_index],
                    color="black",
                    linestyle="-.",
                )

                if iterations and hasattr(self, "_iterations"):
                    axe.plot(
                        self._iterations.reshape(-1, 1),
                        loss(self._iterations.reshape(-1, 1)),
                        linestyle="-",
                        marker="o",
                        color="black",
                        linewidth=0.75,
                        markersize=2,
                    )

                axe.scatter(p0, loss(*p0))

                if hasattr(self, "_minimize"):
                    axe.scatter(
                        self._minimize["parameters"],
                        loss(*self._minimize["parameters"]),
                    )

                if add_labels:
                    axe.set_xlabel(r"Parameter, $\beta_{{{}}}$".format(first_index + 1))
                    label = r"Loss, $\rho(\beta_{{{}}})$".format(first_index + 1)
                    if log_loss:
                        label = "Log-" + label
                    axe.set_ylabel(label)

                axe._pair_indices = (first_index, first_index)

            elif self.k == 2 or (first_index is not None and second_index is not None):
                first_index = first_index or 0
                second_index = second_index or 1

                if first_index == second_index:
                    raise ConfigurationError(
                        "First and second index cannot be the same"
                    )

                x, y = np.meshgrid(scales[first_index], scales[second_index])
                p0 = self._solution["parameters"]
                parameters = list(p0)
                parameters[first_index] = x
                parameters[second_index] = y
                loss = self.parametrized_loss(sigma=self._sigma)(*parameters)

                if log_loss:
                    loss = np.log10(loss)
                    ploss = np.log10(self._loss)
                else:
                    ploss = self._loss

                if surface:
                    # 3D Surfaces:
                    axe.plot_surface(
                        x, y, loss, cmap="jet", rstride=1, cstride=1, alpha=0.50
                    )
                    axe.contour(
                        x, y, loss, zdir="z", offset=ploss, levels=10, cmap="jet"
                    )
                    axe.set_zlabel(r"Loss, $\rho(\beta)$")

                else:
                    # Contours
                    clabels = axe.contour(x, y, loss, levels or 10, cmap="jet")
                    axe.clabel(clabels, clabels.levels, inline=True, fontsize=7)

                    axe.axvline(
                        self._solution["parameters"][first_index],
                        color="black",
                        linestyle="-.",
                    )
                    axe.axhline(
                        self._solution["parameters"][second_index],
                        color="black",
                        linestyle="-.",
                    )

                    if iterations and hasattr(self, "_iteration"):
                        axe.plot(
                            self._iterations[:, first_index].reshape(-1, 1),
                            self._iterations[:, second_index].reshape(-1, 1),
                            linestyle="-",
                            marker="o",
                            color="black",
                            linewidth=0.75,
                            markersize=2,
                        )

                if surface:
                    axe.scatter(p0[first_index], p0[second_index], ploss)
                else:
                    axe.scatter(p0[first_index], p0[second_index])

                if hasattr(self, "_minimize"):
                    axe.scatter(
                        self._minimize["parameters"][first_index],
                        self._minimize["parameters"][second_index],
                    )

                if add_labels:
                    axe.set_xlabel(r"Parameter, $\beta_{{{}}}$".format(first_index + 1))
                    axe.set_ylabel(
                        r"Parameter, $\beta_{{{}}}$".format(second_index + 1)
                    )

            else:
                raise ConfigurationError("Cannot plot loss due to configuration")

            if add_title:
                axe.set_title(full_title, fontdict={"fontsize": 10})

                if not surface:
                    fig.subplots_adjust(top=0.8, left=0.2)
                else:
                    fig.subplots_adjust(top=0.8)

            if not surface and log_x:
                axe.set_xscale("log")

            if not surface and log_y:
                axe.set_yscale("log")

            axe.grid()

            axe._pair_indices = (first_index, second_index)

            return axe

    def plot_loss(
        self,
        mode="lin",
        ratio=0.1,
        xmin=None,
        xmax=None,
        title="",
        levels=None,
        resolution=75,
        iterations=False,
        log_x=False,
        log_y=False,
        log_loss=False,
    ):
        """
        Sketch and plot full loss landscape in order to assess parameters optima convergence and uniqueness.

        .. image:: ../media/figures/FitLossPlotFull.png
            :width: 560
            :alt: Full Fit Loss Plot

        :param mode:
        :param ratio:
        :param xmin:
        :param xmax:
        :param title:
        :param levels:
        :param resolution:
        :return:
        """
        if self.k <= 2:
            axe = self.plot_loss_low_dimension(
                mode=mode,
                ratio=ratio,
                xmin=xmin,
                xmax=xmax,
                title=title,
                levels=levels,
                resolution=resolution,
                iterations=iterations,
                log_x=log_x,
                log_y=log_y,
                log_loss=log_loss,
            )

        else:
            full_title = "Fit Loss Plot: {}\n{}".format(title, self.get_title())

            fig, axes = plt.subplots(
                ncols=self.k - 1,
                nrows=self.k - 1,
                sharex="col",
                sharey="row",
                gridspec_kw={"wspace": 0.05, "hspace": 0.05},
            )

            for i, j in itertools.product(range(self.k), repeat=2):
                if (i < j) and (j <= self.k):
                    axe = axes[j - 1][i]
                    self.plot_loss_low_dimension(
                        first_index=i,
                        second_index=j,
                        axe=axe,
                        mode=mode,
                        ratio=ratio,
                        xmin=xmin,
                        xmax=xmax,
                        title=title,
                        levels=levels,
                        resolution=resolution,
                        iterations=iterations,
                        log_x=log_x,
                        log_y=log_y,
                        log_loss=log_loss,
                        add_labels=False,
                        add_title=False,
                    )

                    if i == 0:
                        axe.set_ylabel(r"$\beta_{{{}}}$".format(j + 1))

                    if j == self.k - 1:
                        axe.set_xlabel(r"$\beta_{{{}}}$".format(i + 1))

                if (i < j) and (j < self.k - 1):
                    axe = axes[i][j]
                    axe.set_axis_off()

            fig.suptitle(full_title, fontsize=10)
            fig.subplots_adjust(top=0.8, left=0.2)

        return axe

    def dataset(self):
        """
        Return experimental data as a DataFrame

        :return: Pandas DataFrame containing all experimental data
        """

        if self.stored(error=True):
            data = pd.DataFrame(self._xdata)
            data.columns = data.columns.map(lambda x: "x%d" % x)

            extra = {"y": self._ydata}
            if self._sigma is not None:
                extra["sy"] = self._sigma

            if self.fitted(error=False):
                extra["yhat"] = self._yhat

            extra = pd.DataFrame(extra)
            data = pd.concat([data, extra], axis=1)

            if self.fitted(error=False):
                data["yerr"] = data["y"] - data["yhat"]
                data["yerrrel"] = data["yerr"] / data["yhat"]
                data["yerrabs"] = np.abs(data["yerr"])
                data["yerrsqr"] = np.power(data["yerr"], 2)

            if self._sigma is not None and self.fitted(error=False):
                data["chi2"] = ((data["y"] - data["yhat"]) / data["sy"]) ** 2

            data.index = data.index.values + 1
            data.index.name = "id"

            return data

    def synthetic_dataset(
        self,
        xdata=None,
        parameters=None,
        dimension=1,
        mode="lin",
        xmin=-1.0,
        xmax=1.0,
        resolution=30,
        sigma=None,
        scale_mode="abs",
        generator=np.random.normal,
        seed=1234,
        **kwargs,
    ):
        target = self.target_dataset(
            xdata=xdata,
            parameters=parameters,
            dimension=dimension,
            mode=mode,
            xmin=xmin,
            xmax=xmax,
            resolution=resolution,
            sigma=sigma,
            scale_mode=scale_mode,
            generator=generator,
            seed=seed,
            **kwargs,
            full_output=True,
        )

        x = target.pop("x")
        data = pd.DataFrame(x)
        data.columns = data.columns.map(lambda x: "x%d" % x)

        p = target.pop("parameters")
        target = pd.DataFrame(target)

        data = pd.concat([data, target], axis=1)
        data.index = data.index.values + 1
        data.index.name = "id"

        return data

    def load(self, file_or_frame, sep=";", store=True):
        """
        Load and store data from frame or CSV file
        :param file_or_frame:
        :param mode:
        :param sep:
        :param store:
        :return:
        """

        if isinstance(file_or_frame, pd.DataFrame):
            data = file_or_frame
        else:
            data = pd.read_csv(file_or_frame, sep=sep)

        subset = data.filter(regex="^x")
        if subset.shape[1] == 0:
            raise ConfigurationError("Data must contains at least one 'x' column")

        subset = data.filter(regex="^y$")
        if subset.shape[1] != 1:
            raise ConfigurationError("Data must contains a single 'y' column")

        if "id" in data.columns:
            data = data.set_index("id")
        else:
            if data.index.name != "id":
                data.index = data.index.values + 1
                data.index.name = "id"

        logger.info("Loaded file '%s' with shape %s" % (file_or_frame, data.shape))

        if store:
            if "sy" in data.columns:
                sigma = data["sy"]
            else:
                sigma = None

            self.store(data.filter(regex="x").values, data["y"], sigma=sigma)

        return data

    def dump(self, file_or_frame, data=None, summary=False):
        """
        Dump dataset into CSV
        :param file_or_frame:
        :param summary:
        :return:
        """
        if data is None:
            if summary:
                data = self.summary()
            else:
                data = self.dataset()
        data.to_csv(file_or_frame, sep=";", index=True)

    def summary(self):
        """
        Add summary row for LaTeX display
        :return: DataFrame
        """
        data = self.dataset()
        data.loc[r""] = data.sum()
        data.iloc[-1, :-5] = None
        data.iloc[-1, 5] = None
        return data
