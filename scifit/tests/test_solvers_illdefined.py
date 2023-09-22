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


class StrangeSOExpCosSolverRegression(GenericLinearRegression):
    factory = illdefined.StrangeSOExpCosSolver
    parameters = np.array([1.0, 1.0, 1.0, 1.0])
    sigma = 0.5

    def test_model_implementation(self):
        pass

    def test_fit_from_synthetic_dataset(self):
        pass

    def test_model_fit_parameters(self):
        pass

    def test_kolmogorov(self):
        pass

    def test_goodness_of_fit(self):
        pass


class StrangeSOExpCosData:
    xdata = np.array(
        [
            [
                0.10009766,
                0.14990234,
                0.19995117,
                0.25,
                0.30004883,
                0.35009766,
                0.39990234,
                0.44995117,
                0.5,
                0.55004883,
                0.60009766,
                0.64990234,
                0.69995117,
                0.75,
                0.80004883,
                0.85009766,
                0.89990234,
                0.94995117,
                1.0,
                1.05004883,
                1.10009766,
                1.14990234,
                1.19995117,
            ]
        ]
    ).T
    ydata = np.array(
        [
            0.91363212,
            -0.13781298,
            -1.06832094,
            -0.10673643,
            -0.16275403,
            -0.48058228,
            -2.00849513,
            -1.53031754,
            -1.37471773,
            -2.64494386,
            -1.14412059,
            -1.80706345,
            -3.38185233,
            -4.26446956,
            -1.74462464,
            -4.50088441,
            -0.96261029,
            -7.16421132,
            -2.38919002,
            0.066374,
            -1.64572445,
            -7.46744701,
            -4.48635472,
        ]
    )


class StrangeSOExpCosSolverRegressionDataset(
    StrangeSOExpCosData, StrangeSOExpCosSolverRegression, TestCase
):
    configuration = {"p0": np.array([-5.0, 1.0, 0.15, 1.0])}


class StrangeSOExpCosSolverRegressionSynthetic(
    StrangeSOExpCosSolverRegression, TestCase
):
    parameters = np.array([-8.0, 0.1, 0.2, 1.0])
    configuration = {"p0": [1.0, 1.0, 0.5, 1.0]}
    xmin = 0.2
    xmax = 1.2
    resolution = 50
    sigma = 0.1


class SOExpCosSolverRegressionDataset(
    StrangeSOExpCosData, StrangeSOExpCosSolverRegression, TestCase
):
    factory = illdefined.SOExpCosSolver
    parameters = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 1.0])


class SOExpCosSolverRegressionSynthetic(StrangeSOExpCosSolverRegression, TestCase):
    factory = illdefined.SOExpCosSolver
    parameters = np.array([-3.0, 1.0, 0.2, 1.35, 5.0, 0.0])
    configuration = {"p0": np.array([1.0, 1.0, 1.0, 1.0, 6.0, 1.0])}
    xmin = 0.2
    xmax = 1.2
    resolution = 75
    sigma = 0.05
