from unittest import TestCase

import numpy as np
import pandas as pd

from scifit.solvers import kinetics
from scifit.tests.helpers import GenericTestFitSolver
from scifit.tests.test_solvers_linear import GenericLinearRegression


class SimpleKineticRegression(GenericLinearRegression):
    factory = kinetics.SimpleKineticSolver
    parameters = np.array([3.1])
    xmin = 0.0
    xmax = 1000.0
    resolution = 2000


class SimpleKineticRegressionNoiseL0(SimpleKineticRegression, TestCase):
    sigma = 1e-6

