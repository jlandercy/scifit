from unittest import TestCase

import numpy as np
import pandas as pd

from scifit.solvers import kinetics
from scifit.tests.helpers import GenericTestFitSolver
from scifit.tests.test_solvers_linear import GenericLinearRegression


class GenericKineticRegression(GenericTestFitSolver):

    def test_dataset_serialization_equivalence(self):
        pass

    def test_fit_from_synthetic_dataset(self):
        pass

    def test_model_solve_signature_sigma(self):
        pass

    def test_kolmogorov(self):
        pass


class SimpleKineticRegression(GenericKineticRegression):
    factory = kinetics.SimpleKineticSolver
    parameters = np.array([2.1e-2])
    configuration = {"p0": np.array([1e-3])}
    xmin = 0.0
    xmax = 500.0
    resolution = 50


class SimpleKineticRegressionNoiseL0(SimpleKineticRegression, TestCase):
    sigma = 1e-6


class SimpleKineticRegressionNoiseL1(SimpleKineticRegression, TestCase):
    sigma = 2.5e-2


class SimpleKineticRegressionNoiseL2(SimpleKineticRegression, TestCase):
    sigma = 1e-1


class SequenceKineticRegression(GenericKineticRegression):
    factory = kinetics.SequenceKineticSolver
    parameters = np.array([2.1e-2, 3.15e-3])
    configuration = {"p0": np.array([2e-2, 2e-3])}
    xmin = 0.0
    xmax = 1000.0
    resolution = 50


class SequenceKineticRegressionNoiseL0(SequenceKineticRegression, TestCase):
    sigma = 1e-6


class SequenceKineticRegressionNoiseL1(SequenceKineticRegression, TestCase):
    sigma = 2.5e-2


class SequenceKineticRegressionNoiseL2(SequenceKineticRegression, TestCase):
    sigma = 1e-1

