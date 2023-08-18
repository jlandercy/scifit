"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`scifit.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

from scifit.solvers import specials
from scifit.tests.helpers import GenericTestFitSolver
from scifit.tests.test_solvers_linear import GenericLinearRegression


class DebyeRegression(GenericLinearRegression):
    factory = specials.DebyeFitSolver
    parameters = np.array([428.])
    sigma = None
    xmin = 25.
    xmax = 550.


class DebyeRegressionNoiseL0(DebyeRegression, TestCase):
    sigma = 1e-6


class DebyeRegressionNoiseL1(DebyeRegression, TestCase):
    sigma = 1e-2

