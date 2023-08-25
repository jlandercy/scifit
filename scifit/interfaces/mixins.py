import abc
import numbers
from collections.abc import Iterable

import numpy as np
import pandas as pd

from scifit import logger
from scifit.errors.base import *


class ConfigurationMixin(abc.ABC):

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


class FitSolverMixin(ConfigurationMixin):

    _dimension = None
    _data_keys = ("_xdata", "_ydata", "_sigma")
    _result_keys = tuple()

    def __init__(self, dimension=None, *args, **kwargs):
        self._dimension = dimension or self._dimension
        if self._dimension is None:
            raise ConfigurationError("Dimension must be set.")
        if not isinstance(self._dimension, numbers.Integral):
            raise ConfigurationError("Dimension must be an integral number, got %s instead" % type(self._dimension))
        if self._dimension < 0:
            raise ConfigurationError("Dimension must be a positive number.")
        super().__init__(*args, **kwargs)

    def clean_results(self):
        for key in self._result_keys:
            self.__dict__.pop(key, None)

    def store(self, xdata=None, ydata=None, sigma=None, data=None):
        """
        Validate and store features (variables), target and target uncertainties.

        :param xdata: Features (variables) as a :code:`(n,m)` matrix
        :param ydata: Target as a :code:`(n,)` matrix
        :param sigma: Target uncertainties as :code:`(n,)` matrix
        :param data: Full dataset including all xdata, ydata and sigma at once
        :raise: Exception :class:`scifit.errors.base.InputDataError` if validation fails
        """

        # Partially import dataframe if provided, override with other fields
        if data is not None:
            if not isinstance(data, pd.DataFrame):
                raise InputDataError(
                    "Data must be of type DataFrame, got %s instead" % type(data)
                )
            if xdata is None:
                xdata = data.filter(regex="^x[\d]+").values
            if ydata is None:
                ydata = data["y"].values
            if sigma is None and "sy" in data.columns:
                sigma = data["sy"].values

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

        self._xdata = xdata
        self._ydata = ydata
        self._sigma = sigma

        logger.info(
            "Stored data into solver: X%s, y%s, s=%s"
            % (xdata.shape, ydata.shape, sigma is not None)
        )


    # def clean_data(self):
    #     """Clean data"""
    #     pass
    #
    # def clean_attributes(self):
    #     """Clean fitted attributes"""
    #     pass
    #
    # def clean(self):
    #     self.clean_data()
    #     self.clean_attributes()
    #
    # def store(self, xdata=None, ydata=None, sigma=None, **kwargs):
    #     """Store data"""
    #     self.clean()
    #     xdata, ydata, sigma = self.validate(xdata=xdata, ydata=ydata, sigma=sigma)
    #
    # def merge(self, xdata=None, ydata=None, sigma=None, **kwargs):
    #     """Update data"""
    #     pass
    #
    # @staticmethod
    # def create_dataset(xdata=None, ydata=None, sigma=None):
    #     """Assemble standardized dataset from matrix and vectors"""
    #     pass
    #
    # @staticmethod
    # def split_dataset(data=None):
    #     """Split standardized dataset into matrix and vectors"""
    #     pass
    #
    # @classmethod
    # def validate(cls, xdata=None, ydata=None, sigma=None):
    #     """Validate matrix and vectors"""
    #     if isinstance(xdata, pd.DataFrame) and ydata is None:
    #         xdata, ydata, sigma = cls.split_dataset(xdata)
    #
    #     return xdata, ydata, sigma
    #
    # def dataset(self):
    #     """Get data as a standardized frame"""
    #     data = self.create_dataset(xdata=self.xdata, ydata=self.ydata, sigma=self.sigma)
    #     return data
    #
    # @abc.abstractmethod
    # def _fit(self, xdata=None, ydata=None, sigma=None, data=None, **kwargs):
    #     """Class defined fit method"""
    #     pass
    #
    # def fit(self, xdata=None, ydata=None, sigma=None, data=None, **kwargs):
    #     """Fit using data"""
    #     self.clean()
    #     self._fit(xdata=None, ydata=None, sigma=None, data=None, **kwargs)
    #     # Fit
    #     return self
    #
    # def refit(self, xdata=None, ydata=None, sigma=None, data=None, **kwargs):
    #     """Merge data and fit again"""
    #     data = self._merge(xdata=xdata, ydata=ydata, sigma=sigma, data=data)
    #     self.fit(data=data, **kwargs)
    #     return self
