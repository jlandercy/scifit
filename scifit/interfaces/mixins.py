import abc
import inspect
import numbers
from collections.abc import Iterable

import numpy as np
import pandas as pd

from scifit import logger
from scifit.errors.base import *


class ConfigurationMixin:
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


class FitSolverMixin(ConfigurationMixin, abc.ABC):
    _dimension = None
    _data_keys = ("_xdata", "_ydata", "_sigma")
    _result_keys = tuple()
    _model_equation = None
    _equation_array = False

    def __init__(self, dimension=None, *args, **kwargs):
        self._dimension = dimension or self._dimension
        if self._dimension is None:
            raise ConfigurationError("Dimension must be set.")
        if not isinstance(self._dimension, numbers.Integral):
            raise ConfigurationError(
                "Dimension must be an integral number, got %s instead"
                % type(self._dimension)
            )
        if self._dimension < 0:
            raise ConfigurationError("Dimension must be a positive number.")
        super().__init__(*args, **kwargs)

    def stored(self, error=False):
        """
        Boolean switch to indicate if the solver object has stored input data successfully

        :param error: raise error instead of returning
        :return: Stored status as boolean
        :raise: Exception :class:`scifit.errors.base.NotStoredError` if check fails
        """
        is_cleaned = not (hasattr(self, "_xdata") or hasattr(self, "_ydata"))
        if not is_cleaned and error:
            NotCleanedError("Input data are not cleaned")
        return is_cleaned

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

    def parameters(self):
        """Return parameters and uncertainty as a frame"""
        if self.fitted(error=True):
            data = pd.DataFrame(
                {
                    "b": self._solution["parameters"],
                    "sb": np.sqrt(np.diagonal(self._solution["covariance"])),
                }
            )
            return data

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
        return self._dimension or self._xdata.shape[1]

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

    def clean_data(self):
        for key in self._data_keys:
            self.__dict__.pop(key, None)

    def clean_results(self):
        for key in self._result_keys:
            self.__dict__.pop(key, None)

    def clean(self):
        self.clean_results()
        self.clean_data()

    def load(self, file_or_frame, sep=";"):
        """
        Load and store data from frame or CSV file
        :param file_or_frame:
        :param mode:
        :param sep:
        :param store:
        :return:
        """

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

        return data

    def dump(self, file_or_frame, data=None, summary=False):
        """
        Dump dataset into CSV
        :param file_or_frame:
        :param data:
        :param summary:
        :return:
        """
        if data is None:
            data = self.dataset()
        if summary:
            data = self.summary(data=data)
        data.to_csv(file_or_frame, sep=";", index=True)

    def validate(self, xdata=None, ydata=None, sigma=None):
        xdata = np.array(xdata)
        ydata = np.array(ydata)

        if xdata.ndim != 2:
            raise InputDataError("Features must be a two dimensional array")

        if xdata.shape[0] != ydata.shape[0]:
            raise InputDataError(
                "Incompatible shapes between x %s and y %s" % (xdata.shape, ydata.shape)
            )

        if xdata.shape[1] != self.m:
            raise InputDataError(
                "Incompatible second dimension for x, expected %d, got %d instead"
                % (self.m, xdata[1])
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
                raise InputDataError(
                    "Sigma as array must have the same shape as ydata %s, got %s instead"
                    % (ydata.shape, sigma.shape)
                )
            if not np.issubdtype(sigma.dtype, np.number):
                raise InputDataError(
                    "All sigma must be numbers, got '%s' instead: %s"
                    % (sigma.dtype, sigma)
                )
            if not np.all(np.isfinite(sigma)):
                raise InputDataError("All sigma must be finite numbers: %s" % sigma)
            if not np.all(sigma > 0.0):
                raise InputDataError(
                    "All sigma must be strictly positive numbers: %s" % sigma
                )
        elif isinstance(sigma, numbers.Number):
            if sigma <= 0.0:
                raise InputDataError("Sigma must be strictly positive")
        elif sigma is None:
            pass
        else:
            raise InputDataError("Sigma must be a number or an array of number")

        return xdata, ydata, sigma

    def defaults(self, xdata=None, ydata=None, sigma=None, parameters=None):
        if xdata is None:
            xdata = self._xdata
        if ydata is None:
            ydata = self._ydata
        if sigma is None:
            sigma = self._sigma
        if parameters is None:
            if self.fitted():
                parameters = self._solution["parameters"]
            else:
                parameters = np.full((self.k,), 1.0)
        return xdata, ydata, sigma, parameters

    def _store(self, xdata=None, ydata=None, sigma=None):
        """
        Validate and store features (variables), target and target uncertainties.

        :param xdata: Features (variables) as a :code:`(n,m)` matrix
        :param ydata: Target as a :code:`(n,)` matrix
        :param sigma: Target uncertainties as :code:`(n,)` matrix
        :param data: Full dataset including all xdata, ydata and sigma at once
        :raise: Exception :class:`scifit.errors.base.InputDataError` if validation fails
        """

        if xdata is not None and ydata is None:
            if not isinstance(xdata, pd.DataFrame):
                raise InputDataError(
                    "Data must be of type DataFrame, got %s instead" % type(xdata)
                )
            xdata, ydata, sigma = self.split_dataset(xdata)

        # Validate data:
        xdata, ydata, sigma = self.validate(xdata=xdata, ydata=ydata, sigma=sigma)

        self._xdata = xdata
        self._ydata = ydata
        self._sigma = sigma

        logger.info(
            "Stored data into solver: X%s, y%s, s=%s"
            % (xdata.shape, ydata.shape, sigma is not None)
        )

    @abc.abstractmethod
    def _fit(self, xdata=None, ydata=None, sigma=None, **kwargs):
        """Class defined fit method"""
        pass

    def fit(self, xdata=None, ydata=None, sigma=None, **kwargs):
        """Clean old data, store data and fit data to model"""
        self.clean()
        self._store(xdata=xdata, ydata=ydata, sigma=sigma)
        solution = self._fit(xdata=xdata, ydata=ydata, sigma=sigma, **kwargs)
        return solution

    # def refit(self, xdata=None, ydata=None, sigma=None, **kwargs):
    #     """Merge data and fit again"""
    #     data = self._merge(xdata=xdata, ydata=ydata, sigma=sigma, data=data)
    #     self.fit(data=data, **kwargs)
    #     return self

    def predict(self, xdata=None, parameters=None):
        """
        Predict target wrt features (variables) and parameters.
        If parameters are not provided uses regressed parameters (problem needs to be solved first).

        :param xdata: Features (variables) as :code:`(n,m)` matrix
        :param parameters: Sequence of :code:`k` parameters
        :return: Predicted target as a :code:`(n,)` matrix
        """
        xdata, _, _, parameters = self.defaults(
            xdata=xdata, ydata=False, sigma=False, parameters=parameters
        )
        return self.model(xdata, *parameters)

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
        resolution=100,
    ):
        """
        Generate features scales
        """
        if domains is None and self.stored(error=False):
            domains = self.feature_domains()
        return self.scales(
            domains=domains,
            mode=mode,
            xmin=xmin,
            xmax=xmax,
            dimension=self.m,
            resolution=resolution,
        )

    def feature_space(
        self,
        domains=None,
        mode="lin",
        xmin=None,
        xmax=None,
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
                resolution=resolution,
            )
        )

    def feature_dataset(
        self,
        domains=None,
        mode="lin",
        xmin=None,
        xmax=None,
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
        ratio=None,
        factor=None,
        precision=1e-9,
        include_origin=False,
        include_unit=False,
        iterations=False,
    ):
        """
        Generate parameter domains, useful for drawing scales fitting the parameters space
        """

        if mode not in {"lin", "log"}:
            raise ConfigurationError(
                "Domain mode must be in {lin, log} got '%s' instead" % mode
            )

        ratio = ratio or (0.1 if mode == "lin" else 10.0)
        factor = factor or (5.0 if mode == "lin" else 2.0)

        if iterations:
            domains = pd.DataFrame(self._iterations).describe()
            domains.loc["extent", :] = domains.loc["max", :] - domains.loc["min", :]
            domains.loc["min", :] -= ratio * domains.loc["extent", :]
            domains.loc["max", :] += ratio * domains.loc["extent", :]
            domains = domains.loc[["min", "max"], :]

        else:
            if parameters is None and self.fitted(error=True):
                parameters = self._solution["parameters"]

            if parameters is not None:
                if mode == "lin":
                    xmin = xmin or list(
                        parameters - factor * ratio * np.abs(parameters)
                    )
                    xmax = xmax or list(
                        parameters + factor * ratio * np.abs(parameters)
                    )

                elif mode == "log":
                    xmin = xmin or list(parameters / np.power(ratio, factor))
                    xmax = xmax or list(parameters * np.power(ratio, factor))

            xmin = xmin or precision
            if not isinstance(xmin, Iterable):
                xmin = [xmin] * self.k

            xmin = np.array(xmin)
            if include_origin:
                xmin[xmin >= 0.0] = 0.0

            if len(xmin) != self.k:
                raise ConfigurationError(
                    "Domain lower boundaries must have the same dimension as parameter space"
                )

            xmax = xmax or 1.0
            if not isinstance(xmax, Iterable):
                xmax = [xmax] * self.k

            xmax = np.array(xmax)
            if include_unit:
                xmax[xmax <= 1.0] = 1.0

            if len(xmax) != self.k:
                raise ConfigurationError(
                    "Domain upper boundaries must have the same dimension as parameter space"
                )

            domains = pd.DataFrame([xmin, xmax], index=["min", "max"])

        return domains

    def parameter_scales(
        self,
        domains=None,
        mode="lin",
        xmin=None,
        xmax=None,
        ratio=None,
        factor=None,
        include_origin=False,
        include_unit=False,
        resolution=100,
    ):
        """
        Generate parameter scales
        """
        if domains is None:
            domains = self.parameter_domains(
                mode=mode,
                xmin=xmin,
                xmax=xmax,
                ratio=ratio,
                factor=factor,
                include_origin=include_origin,
                include_unit=include_unit,
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

    @staticmethod
    def split_dataset(data):
        """Split dataset into matrix and vectors"""
        if not isinstance(data, pd.DataFrame):
            raise InputDataError(
                "Data must be of type DataFrame, got %s instead" % type(data)
            )
        xdata = data.filter(regex="^x[\d]+").values
        ydata = data["y"].values
        if "sy" in data.columns:
            sigma = data["sy"].values
        else:
            sigma = None
        return xdata, ydata, sigma

    @staticmethod
    def merge_dataset(xdata, ydata, sigma=None):
        """
        Merge matrix and vectors into dataset
        :param xdata:
        :param ydata:
        :param sigma:
        :return:
        """

        data = pd.DataFrame(xdata)
        data.columns = data.columns.map(lambda x: "x%d" % x)

        extra = {"y": ydata}
        if sigma is not None:
            extra["sy"] = sigma

        extra = pd.DataFrame(extra)
        data = pd.concat([data, extra], axis=1)

        return data

    def synthetic_dataset(
        self,
        xdata=None,
        parameters=None,
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

    def dataset(self):
        """
        Return experimental data as a DataFrame

        :return: Pandas DataFrame containing all experimental data
        """

        if self.stored(error=True):
            data = self.merge_dataset(self._xdata, self._ydata, self._sigma)

            if self.fitted(error=False):
                data["yhat"] = self._yhat
                data["yerr"] = data["y"] - data["yhat"]
                data["yerrrel"] = data["yerr"] / data["yhat"]
                data["yerrabs"] = np.abs(data["yerr"])
                data["yerrsqr"] = np.power(data["yerr"], 2)

            if self._sigma is not None and self.fitted(error=False):
                data["chi2"] = ((data["y"] - data["yhat"]) / data["sy"]) ** 2

            data.index = data.index.values + 1
            data.index.name = "id"

            return data

    def summary(self, data=None):
        """
        Add summary row for LaTeX display
        :param data:
        :return: DataFrame
        """
        if data is None:
            data = self.dataset()
        data.loc[r""] = data.sum()
        data.iloc[-1, :-5] = None
        data.iloc[-1, 5] = None
        return data
