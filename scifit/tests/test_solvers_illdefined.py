from unittest import TestCase

import numpy as np

from scifit.solvers import illdefined, linear, scientific
from scifit.tests.helpers import GenericTestFitSolver
from scifit.tests.test_solvers_linear import GenericLinearRegression


class IncompleteLogisticRegression(GenericLinearRegression):
    factory = scientific.LogisticFitSolver
    parameters = np.array([3.1, 1.27])


class IncompleteLogisticRegressionNoiseL0(IncompleteLogisticRegression, TestCase):
    sigma = 1e-6
