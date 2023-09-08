import itertools
import warnings

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy import optimize, stats

from scifit import logger
from scifit.errors.base import *
from scifit.interfaces.mixins import *
from scifit.toolbox.report import FitSolverReportProcessor


class FitSolverInterface(FitSolverMixin):

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

    _dimension = None
    _data_keys = ("_xdata", "_ydata", "_sigma")
    _result_keys = (
        "_solution",
        "_minimize",
        "_yhat",
        "_loss",
        "_score",
        "_gof",
        "_k2s",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        kwargs = self.configuration(**kwargs)

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
            **kwargs,
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

    def minimize(self, xdata, ydata, sigma=None, **kwargs):
        """
        Solve the fitting problem by finding a set of parameters minimizing the loss function wrt features, target and sigma.
        Return structured solution and update solver object in order to expose analysis convenience (fit, loss).

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param ydata: Target as :code:`(n,)` matrix
        :param sigma: Uncertainty on target as :code:`(n,)` matrix or scalar or :code:`None`
        :param kwargs: Extra parameters to pass to :code:`scipy.optimize.curve_fit`
        :return: Dictionary of objects with details about the regression including regressed parameters and final covariance
        """

        kwargs = self.configuration(**kwargs)

        # Adapt default parameters as curve_fit
        p0 = kwargs.pop("p0", None)
        if p0 is None:
            p0 = np.full((self.k,), 1.0)

        bounds = kwargs.pop("bounds", None)
        if isinstance(bounds, Iterable):
            bounds = [(a, b) for a, b in zip(*bounds)]

        # Adapt signature for single number:
        if isinstance(sigma, numbers.Number):
            sigma = np.full(ydata.shape, sigma)

        # Adapt loss function signature:
        def loss(p):
            return self.parametrized_loss(xdata, ydata, sigma=sigma)(*p)

        def callback(result):
            self._iterations.append(result)
            self.minimize_callback(result)

        self._iterations = [p0]
        solution = optimize.minimize(
            loss,
            x0=p0,
            method="L-BFGS-B",
            jac="3-point",
            callback=callback,
            bounds=bounds,
            **kwargs,
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.sum(
                wdata
                * np.power((ydata - self.predict(xdata, parameters=parameters)), 2)
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

    def score(self, xdata, ydata, sigma=None, parameters=None, estimated=False):
        """
        Compute Coefficient of Determination :math:`R^2` as follows:

        .. math::

            R^2 = 1 - \\frac{RSS}{TSS}

        Or in its estimated form (default):

        .. math::

            R^2 = 1 - \\frac{RSS / d_\\nu}{TSS / d_n}

        Where:

        - :math:`d_\\nu = n - k - 1` is the degree of freedom for Residual Sum of Squared of the model;
        - :math:`d_\\nu = n - 1` is the degree of freedom of the Variance

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param ydata: Target as :code:`(n,)` matrix
        :param sigma: Uncertainty on target as :code:`(n,)` matrix or scalar or :code:`None`
        :param parameters: Sequence of :code:`k` parameters
        :param estimated:
        :return: Coefficient of Determination :math:`R^2`
        """

        score = 1.0 - self.RSS(xdata, ydata, parameters=parameters) / self.TSS(ydata)

        if estimated:
            score *= (self.n - 1) / (self.n - self.k - 1)

        return score

    score.name = "$R^2$"

    def goodness_of_fit(self):
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
        if self.fitted(error=True):
            statistic = self.chi_square(
                self._xdata, self._ydata, sigma=self._sigma, parameters=None
            )
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
                    {"alpha": alpha, "low-value": chi, "H0": chi <= statistic}
                )
                # Right Sided Test:
                chi = law.ppf(1.0 - alpha)
                result["significance"]["right-sided"].append(
                    {"alpha": alpha, "high-value": chi, "H0": statistic <= chi}
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

    def kolmogorov(self):
        if self.fitted(error=True):
            test = stats.ks_2samp(
                self._yhat, self._ydata, alternative="two-sided", method="asymp"
            )
            return {
                "statistic": test.statistic,
                "pvalue": test.pvalue,
                "test": test,
            }

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

        xdata, ydata, sigma, _ = self.defaults(xdata=xdata, ydata=ydata, sigma=sigma)

        @np.vectorize
        def wrapped(*parameters):
            return self.loss(xdata, ydata, sigma=sigma, parameters=parameters)

        return wrapped

    def _fit(self, xdata=None, ydata=None, sigma=None, **kwargs):
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

        # Solve the Adjustment problem:
        self._solution = self.solve(
            self._xdata, self._ydata, sigma=self._sigma, **kwargs
        )

        # Solve again the regression problem in a different way to check the regression and parameters pathway:
        self._minimize = self.minimize(
            self._xdata, self._ydata, sigma=self._sigma, **kwargs
        )

        # Estimates & metrics:
        self._yhat = self.predict(self._xdata)
        self._loss = self.loss(self._xdata, self._ydata, sigma=self._sigma)
        self._score = self.score(self._xdata, self._ydata, sigma=self._sigma)

        # Statistical tests:
        self._gof = self.goodness_of_fit()
        self._k2s = self.kolmogorov()

        return self._solution

    def get_latex_parameters(self, show_sigma=True, precision=3, mode="g"):
        """
        Return parameters in a compact LaTeX fashion, useful for figure title
        """
        if self.fitted(error=True):
            terms = []
            for i, parameter in enumerate(self._solution["parameters"]):
                term = ("{:.%d%s}" % (precision, mode)).format(parameter)
                if show_sigma:
                    term += (" \xb1 {:.%d%s}" % (precision, mode)).format(
                        np.sqrt(self._solution["covariance"][i][i])
                    )
                terms.append(term)
            return r"$\beta \pm s_{{\beta}}$ = ({})".format("; ".join(terms))

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
        resolution=250,
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

        if not title and self._model_equation and not self._equation_array:
            title = "$%s$" % self._model_equation

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

    def plot_chi_square(self, title="", resolution=250):
        """
        Plot Chi Square Goodness of Fit figure, summarizes all critical thresholds and p-value

        .. image:: ../media/figures/GoodnessOfFitPlot.png
            :width: 560
            :alt: Chi Square Goodness of Fit Plot

        :param title:
        :param resolution:
        :return:
        """

        if not title and self._model_equation and not self._equation_array:
            title = "$%s$" % self._model_equation

        if self.fitted(error=True):
            full_title = "Fit $\chi^2$ Plot: {}\n{}".format(title, self.get_title())

            law = self._gof["law"]
            statistic = self._gof["statistic"]
            xmin = min(law.ppf(0.0001), statistic - 1)
            xmax = max(law.ppf(0.9999), statistic + 1)
            xlin = np.linspace(xmin, xmax, resolution)
            xarea = np.linspace(statistic, xmax, resolution)

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

    def plot_kolmogorov(self, title=""):
        """
        Plot Kolmogorov Goodness of Fit figure

        .. image:: ../media/figures/KolmogorovPlot.png
            :width: 560
            :alt: Chi Square Goodness of Fit Plot

        :param title:
        :return:
        """

        if not title and self._model_equation and not self._equation_array:
            title = "$%s$" % self._model_equation

        if self.fitted(error=True):
            full_title = "Fit Kolmogorov Plot: {}\n{}".format(title, self.get_title())

            fig, axe = plt.subplots()

            test = self.kolmogorov()["test"]

            y_ecdf = stats.ecdf(self._ydata)
            y_ecdf.cdf.plot(axe)

            yhat_ecdf = stats.ecdf(self._yhat)
            yhat_ecdf.cdf.plot(axe)

            axe.axvline(test.statistic_location, linestyle="-.", color="black")

            axe.set_title(full_title, fontdict={"fontsize": 10})
            axe.set_xlabel(r"Target, $y$")
            axe.set_ylabel(r"ECDF, $F(y)$")

            axe.legend(
                [
                    r"Data: $F_1(y)$",
                    r"Fit: $F_2(\hat{y})$",
                    (
                        r"K-Test: $D({:.3g}) = {:.3g}$".format(
                            test.statistic_location, test.statistic
                        )
                        + "\n"
                        + r"$p = P(K > D) = {:.4f}$".format(test.pvalue)
                    ),
                ]
            )
            axe.grid()

            fig.subplots_adjust(top=0.8, left=0.15)

            return axe

    def plot_loss_low_dimension(
        self,
        first_index=None,
        second_index=None,
        axe=None,
        mode="lin",
        domains=None,
        ratio=None,
        factor=None,
        xmin=None,
        xmax=None,
        title="",
        levels=None,
        resolution=100,
        surface=False,
        iterations=False,
        include_origin=False,
        include_unit=False,
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

        .. image:: ../media/figures/FitLossSurfacePlot.png
            :width: 560
            :alt: Low Dimensionality Fit Loss Plot (surface)

        See :meth:`FitSolverInterface.plot_loss` for full loss landscape.
        """

        if not title and self._model_equation and not self._equation_array:
            title = "$%s$" % self._model_equation

        if self.fitted(error=True):
            if axe is None:
                if surface:
                    fig, axe = plt.subplots(subplot_kw={"projection": "3d"})
                else:
                    fig, axe = plt.subplots()
            fig = axe.figure

            full_title = "Fit {}Loss Plot: {}\n{}".format(
                "Log-" if log_loss else "", title, self.get_title()
            )

            scales = self.parameter_scales(
                mode=mode,
                domains=domains,
                ratio=ratio,
                factor=factor,
                xmin=xmin,
                xmax=xmax,
                resolution=resolution,
                include_origin=include_origin,
                include_unit=include_unit,
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
                    xiter = self._iterations.reshape(-1, 1)
                    axe.plot(
                        xiter,
                        loss(xiter),
                        linestyle=":",
                        marker="o",
                        color="darkgray",
                        # linewidth=0.75,
                        markersize=4,
                    )

                axe.scatter(p0, loss(*p0))

                if iterations and hasattr(self, "_minimize"):
                    axe.scatter(
                        self._minimize["parameters"],
                        loss(*self._minimize["parameters"]),
                    )

                if add_labels:
                    axe.set_xlabel(r"Parameter, $\beta_{{{}}}$".format(first_index + 1))
                    label = r"{}Loss, $\rho(\beta_{{{}}})$".format(
                        "Log-" if log_loss else "", first_index + 1
                    )
                    axe.set_ylabel(label)

                if domains is not None:
                    axe.set_xlim(domains.loc[["min", "max"], first_index])

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
                        x,
                        y,
                        loss,
                        zdir="z",
                        offset=ploss,
                        levels=levels or 10,
                        cmap="jet",
                    )
                    axe.set_zlabel(
                        r"{}Loss, $\rho(\beta)$".format("Log-" if log_loss else "")
                    )

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

                    if iterations and hasattr(self, "_iterations"):
                        axe.plot(
                            self._iterations[:, first_index].reshape(-1, 1),
                            self._iterations[:, second_index].reshape(-1, 1),
                            linestyle=":",
                            marker="o",
                            color="darkgray",
                            # linewidth=0.75,
                            markersize=4,
                        )

                if surface:
                    axe.scatter(p0[first_index], p0[second_index], ploss)
                else:
                    axe.scatter(p0[first_index], p0[second_index])

                if iterations and hasattr(self, "_minimize"):
                    axe.scatter(
                        self._minimize["parameters"][first_index],
                        self._minimize["parameters"][second_index],
                    )

                if add_labels:
                    axe.set_xlabel(r"Parameter, $\beta_{{{}}}$".format(first_index + 1))
                    axe.set_ylabel(
                        r"Parameter, $\beta_{{{}}}$".format(second_index + 1)
                    )

                if domains is not None:
                    axe.set_xlim(domains.loc[["min", "max"], first_index])
                    axe.set_ylim(domains.loc[["min", "max"], second_index])

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
        domains=None,
        ratio=None,
        factor=None,
        xmin=None,
        xmax=None,
        title="",
        levels=None,
        resolution=100,
        iterations=False,
        include_origin=False,
        include_unit=False,
        log_x=False,
        log_y=False,
        log_loss=False,
        scatter=True,
        surface=False,
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

        if not title and self._model_equation and not self._equation_array:
            title = "$%s$" % self._model_equation

        if self.k <= 2:
            axes = self.plot_loss_low_dimension(
                mode=mode,
                domains=domains,
                ratio=ratio,
                factor=factor,
                xmin=xmin,
                xmax=xmax,
                title=title,
                levels=levels,
                resolution=resolution,
                iterations=iterations,
                include_origin=include_origin,
                include_unit=include_unit,
                log_x=log_x,
                log_y=log_y,
                log_loss=log_loss,
                surface=surface,
            )

        elif scatter and not surface:
            full_title = "Fit {}Loss Plot: {}\n{}".format(
                "Log-" if log_loss else "", title, self.get_title()
            )

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
                        domains=domains,
                        ratio=ratio,
                        factor=factor,
                        xmin=xmin,
                        xmax=xmax,
                        title=title,
                        levels=levels,
                        resolution=resolution,
                        iterations=iterations,
                        include_origin=include_origin,
                        include_unit=include_unit,
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

        else:
            axes = []
            for i, j in itertools.combinations(range(self.k), 2):
                axe = self.plot_loss_low_dimension(
                    title=title,
                    first_index=i,
                    second_index=j,
                    mode=mode,
                    domains=domains,
                    ratio=ratio,
                    factor=factor,
                    xmin=xmin,
                    xmax=xmax,
                    levels=levels,
                    resolution=resolution,
                    iterations=iterations,
                    include_origin=include_origin,
                    include_unit=include_unit,
                    log_x=log_x,
                    log_y=log_y,
                    log_loss=log_loss,
                    surface=surface,
                )
                axes.append(axe)

        return axes

    def chi_square_table(self):
        if self.fitted(error=True):
            frames = []
            for key in ["left-sided", "two-sided", "right-sided"]:
                frames.append(
                    pd.DataFrame(self._gof["significance"][key]).assign(key=key)
                )
            frame = pd.concat(frames, axis=0)
            return frame

    def report(self, file, path=".", mode="pdf"):
        processor = FitSolverReportProcessor()
        processor.report(self, file=file, path=path, mode=mode)


class FitSolver1D(FitSolverInterface):
    _dimension = 1


class FitSolver2D(FitSolverInterface):
    _dimension = 2


class FitSolver3D(FitSolverInterface):
    _dimension = 3
