from unittest import TestCase

import numpy as np
import pandas as pd

from scifit.solvers import illdefined, linear, scientific
from scifit.tests.helpers import GenericTestFitSolver
from scifit.tests.test_solvers_linear import GenericLinearRegression


class IncompleteLogisticRegression(GenericLinearRegression):
    factory = scientific.LogisticFitSolver
    parameters = np.array([3.1, 1.27])


class IncompleteLogisticRegressionNoiseL0(IncompleteLogisticRegression, TestCase):
    sigma = 1e-6


class LinearSquaredSlopeRegression(GenericLinearRegression):
    factory = illdefined.LinearSquaredSlopeSolver
    parameters = np.array([2.0, 1.0])
    loss_domains = pd.DataFrame({"min": [-3, -2], "max": [3, 4]}).T
    log_loss = True


class LinearSquaredSlopeRegressionNoiseL0(LinearSquaredSlopeRegression, TestCase):
    sigma = 1e-6


class LinearSquaredSlopeRegressionNoiseL1(LinearSquaredSlopeRegression, TestCase):
    sigma = 2.5e-2


class LinearSquaredSlopeRegressionNoiseL2(LinearSquaredSlopeRegression, TestCase):
    sigma = 1e-1
