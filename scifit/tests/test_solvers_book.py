from unittest import TestCase

import numpy as np

from scifit.solvers import linear, scientific
from scifit.tests.helpers import GenericTestFitSolver
from scifit.tests.test_solvers_linear import GenericLinearRegression


class BookSampleRegression(GenericLinearRegression):
    root_path = ".cache/media/book/"
    format = "eps"
    fake_sigma_factor = None

    def setUp(self):
        super().setUp()
        if self.fake_sigma_factor is not None:
            self.sigma *= self.fake_sigma_factor
            self.sigmas *= self.fake_sigma_factor


class BookIntroduction2DigitsRegression(BookSampleRegression, TestCase):
    factory = scientific.LogisticFitSolver
    parameters = np.array([1.45, 5.0])
    seed = 12345
    resolution = 30
    sigma = 0.125
    xmin = 0.0
    xmax = 10.0


class BookIntroduction3DigitsRegression(BookIntroduction2DigitsRegression):
    sigma = 0.025
    seed = 123456
    fake_sigma_factor = 1.01


class BookLinearSample01Regression(BookSampleRegression):
    factory = linear.LinearFitSolver
    parameters = np.array([3.0, 2.0])
    resolution = 15


class BookLinearSample01NoiseL0(BookLinearSample01Regression, TestCase):
    sigma = 0.075


class BookLinearSample01NoiseL0Overestimated(BookLinearSample01Regression, TestCase):
    sigma = 0.025
    fake_sigma_factor = 1.45

    def test_goodness_of_fit(self):
        pass


class BookLinearSample01NoiseL0Underestimated(BookLinearSample01Regression, TestCase):
    sigma = 0.025
    fake_sigma_factor = 0.55

    def test_goodness_of_fit(self):
        pass
