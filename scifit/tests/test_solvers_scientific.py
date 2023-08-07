"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`scifit.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

from scifit.solvers import scientific
from scifit.tests.helpers import GenericTestFitSolver
from scifit.tests.test_solvers_linear import GenericLinearRegression


class GompertzRegression(GenericLinearRegression):
    factory = scientific.GompertzFitSolver
    parameters = np.array([1., 1., 2.])
    sigma = None


class GompertzRegressionNoiseL0(GompertzRegression, TestCase):
    sigma = None


class GompertzRegressionNoiseL1(GompertzRegression, TestCase):
    sigma = 1e-3


class GompertzRegressionNoiseL2(GompertzRegression, TestCase):
    sigma = 1e-2


class GompertzRegressionNoiseL3(GompertzRegression, TestCase):
    sigma = 1e-1


class GompertzRegressionNoiseL4(GompertzRegression, TestCase):
    sigma = 1.


# class GompertzRegressionNoiseL5(GompertzRegression, TestCase):
#     sigma = 10.


class GenericKineticRegression(GenericTestFitSolver):
    factory = scientific.MichaelisMentenKineticFitSolver
    parameters = np.array([1., 1.])
    xmin = 1e-6
    xmax = 1e-3
    mode = "lin"
    proportional = True
    sigma = None


class MichaelisMentenKineticRegressionNoiseL0(GenericKineticRegression, TestCase):
    parameters = np.array([4.1e-3, 2.5e-5])
    sigma = None


class MichaelisMentenKineticRegressionNoiseL1(GenericKineticRegression, TestCase):
    parameters = np.array([4.1e-3, 2.5e-5])
    sigma = 1e-3


class MichaelisMentenKineticRegressionNoiseL2(GenericKineticRegression, TestCase):
    parameters = np.array([4.1e-3, 2.5e-5])
    sigma = 1e-2


class MichaelisMentenKineticRegressionNoiseL3(GenericKineticRegression, TestCase):
    parameters = np.array([4.1e-3, 2.5e-5])
    sigma = 1e-1


class MichaelisMentenKineticRegressionNoiseL4(GenericKineticRegression, TestCase):
    parameters = np.array([4.1e-3, 2.5e-5])
    sigma = 1.


class MichaelisMentenKineticRegressionNoiseL5(GenericKineticRegression, TestCase):
    parameters = np.array([4.1e-3, 2.5e-5])
    sigma = 10.
