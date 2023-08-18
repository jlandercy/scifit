"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`scifit.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

from scifit.solvers import specials
from scifit.tests.helpers import GenericTestFitSolver
from scifit.tests.test_solvers_linear import GenericLinearRegression


class XRegression(GenericLinearRegression):
    #factory = scientific.GompertzFitSolver
    parameters = np.array([2.0, 0.5, 5.0])
    sigma = None


class XRegressionNoiseL0(XRegression, TestCase):
    sigma = 1e-6
