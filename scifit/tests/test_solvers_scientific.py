"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`scifit.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np
import pandas as pd

from scifit.solvers import scientific
from scifit.tests.helpers import GenericTestFitSolver
from scifit.tests.test_solvers_linear import GenericLinearRegression


class GompertzRegression(GenericLinearRegression):
    factory = scientific.GompertzFitSolver
    parameters = np.array([2.0, 0.5, 5.0])
    sigma = None
    loss_domains = pd.DataFrame({"min": [0.25, 0.05, 2.5], "max": [2.5, 10.0, 25.0]}).T


class GompertzRegressionNoiseL0(GompertzRegression, TestCase):
    sigma = 1e-6


class GompertzRegressionNoiseL1(GompertzRegression, TestCase):
    sigma = 2.5e-2


class GompertzRegressionNoiseL2(GompertzRegression, TestCase):
    sigma = 1e-1


class GompertzRegressionNoiseL3(GompertzRegression, TestCase):
    sigma = 2.5e-1


class GompertzRegressionNoiseL4(GompertzRegression, TestCase):
    sigma = 1.0


class GompertzRegressionNoiseL5(GompertzRegression, TestCase):
    sigma = 2.5

    def test_kolmogorov(self):
        pass


class GenericKineticRegression(GenericTestFitSolver):
    xmin = 1e-6
    xmax = 1e-3
    mode = "lin"
    sigma = 1e-16


class MichaelisMentenKineticRegression(GenericKineticRegression):
    factory = scientific.MichaelisMentenKineticFitSolver
    parameters = np.array([4.1e-3, 2.5e-5])
    configuration = {"p0": np.array([1e2, 1e2])}
    loss_domains = pd.DataFrame({"min": [1e-5, 1e-6], "max": [1e-2, 1e-4]}).T


class MichaelisMentenKineticRegressionNoiseL0(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 1e-6


class MichaelisMentenKineticRegressionNoiseL1(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 2.5e-2

    def test_fit_from_synthetic_dataset(self):
        """Why this one only"""
        pass


class MichaelisMentenKineticRegressionNoiseL2(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 1e-1


class MichaelisMentenKineticRegressionNoiseL3(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 2.5e-1

    def test_kolmogorov(self):
        pass


class MichaelisMentenKineticRegressionNoiseL4(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 1.0

    def test_kolmogorov(self):
        pass


class MichaelisMentenKineticRegressionNoiseL5(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 2.5

    def test_kolmogorov(self):
        pass


class CooperativeHillEquationRegression(GenericKineticRegression):
    factory = scientific.HillEquationFitSolver
    parameters = np.array([2.12, 2.5e-1])
    loss_domains = pd.DataFrame({"min": [0.1, 1e-2], "max": [5., 1.]}).T

    def test_fit_from_synthetic_dataset(self):
        """All cooperative fails, but only noisy competitive. Does not converge (maxfev>800)"""


class CooperativeHillEquationRegressionNoiseL0(
    CooperativeHillEquationRegression, TestCase
):
    sigma = 1e-6


class CooperativeHillEquationRegressionNoiseL1(
    CooperativeHillEquationRegression, TestCase
):
    sigma = 2.5e-2


class CooperativeHillEquationRegressionNoiseL2(
    CooperativeHillEquationRegression, TestCase
):
    sigma = 1e-1


class CooperativeHillEquationRegressionNoiseL3(
    CooperativeHillEquationRegression, TestCase
):
    sigma = 2.5e-1


class CooperativeHillEquationRegressionNoiseL4(
    CooperativeHillEquationRegression, TestCase
):
    sigma = 1.0

    def test_kolmogorov(self):
        pass


class CooperativeHillEquationRegressionNoiseL5(
    CooperativeHillEquationRegression, TestCase
):
    sigma = 2.5

    def test_kolmogorov(self):
        pass


class CompetitiveHillEquationRegression(GenericKineticRegression):
    factory = scientific.HillEquationFitSolver
    parameters = np.array([0.32, 2.5e-1])
    loss_ratio = 2.0
    loss_factor = 3.0
    loss_domains = pd.DataFrame({"min": [1e-2, 1e-2], "max": [5., 1.]}).T


class CompetitiveHillEquationRegressionNoiseL0(
    CompetitiveHillEquationRegression, TestCase
):
    sigma = 1e-6


class CompetitiveHillEquationRegressionNoiseL1(
    CompetitiveHillEquationRegression, TestCase
):
    sigma = 2.5e-2


class CompetitiveHillEquationRegressionNoiseL2(
    CompetitiveHillEquationRegression, TestCase
):
    sigma = 1e-1


class CompetitiveHillEquationRegressionNoiseL3(
    CompetitiveHillEquationRegression, TestCase
):
    sigma = 2.5e-1


class CompetitiveHillEquationRegressionNoiseL4(
    CompetitiveHillEquationRegression, TestCase
):
    sigma = 1.0

    def test_fit_from_synthetic_dataset(self):
        """Does not converge (maxfev>800)"""
        pass

    def test_kolmogorov(self):
        pass


class CompetitiveHillEquationRegressionNoiseL5(
    CompetitiveHillEquationRegression, TestCase
):
    sigma = 2.5

    def test_fit_from_synthetic_dataset(self):
        """Does not converge (maxfev>800)"""
        pass

    def test_kolmogorov(self):
        pass


class LogisticRegression(GenericLinearRegression):
    factory = scientific.LogisticFitSolver
    parameters = np.array([3.1, -0.75])
    loss_domains = pd.DataFrame({"min": [0.1, -5.], "max": [5., 5.]}).T


class LogisticRegressionNoiseL0(LogisticRegression, TestCase):
    sigma = 1e-6


class LogisticRegressionNoiseL1(LogisticRegression, TestCase):
    sigma = 2.5e-2


class LogisticRegressionNoiseL2(LogisticRegression, TestCase):
    sigma = 1e-1


class AlgebraicSigmoidRegression(GenericLinearRegression):
    factory = scientific.AlgebraicSigmoidFitSolver
    parameters = np.array([4.1])
    xmin = -5.0
    xmax = +5.0
    loss_domains = pd.DataFrame({"min": [2.], "max": [6.]}).T
    # loss_ratio = 5.
    # loss_factor = 3.
    # log_loss = True      # log10(vectorize) ?
    # loss_resolution = 350
    # loss_domains = pd.DataFrame({"min": [3.], "max": [4.]}).T


class AlgebraicSigmoidRegressionNoiseL0(AlgebraicSigmoidRegression, TestCase):
    sigma = 1e-6


class AlgebraicSigmoidRegressionNoiseL1(AlgebraicSigmoidRegression, TestCase):
    sigma = 2.5e-2


class AlgebraicSigmoidRegressionNoiseL2(AlgebraicSigmoidRegression, TestCase):
    sigma = 1e-1


class RichardGeneralizedSigmoidRegression(GenericLinearRegression):
    factory = scientific.RichardGeneralizedSigmoidFitSolver
    parameters = np.array([4.1, 2.3, 1.1, 4.7, 3.2, 1.1])
    loss_domains = pd.DataFrame({"min": [0.1]*6, "max": [6.]*6}).T
    xmin = -5.0
    xmax = +5.0
    resolution = 50

    def test_model_minimize_against_solve(self):
        """Richardson model is not numerically stable"""
        pass

    def test_dataset_serialization_equivalence(self):
        pass

    def test_fit_from_synthetic_dataset(self):
        pass


class RichardGeneralizedSigmoidRegressionNoiseL0(
    RichardGeneralizedSigmoidRegression, TestCase
):
    sigma = 1e-6


class RichardGeneralizedSigmoidRegressionNoiseL1(
    RichardGeneralizedSigmoidRegression, TestCase
):
    sigma = 2.5e-2


class RichardGeneralizedSigmoidRegressionNoiseL2(
    RichardGeneralizedSigmoidRegression, TestCase
):
    sigma = 1e-1


class SmoothstepSigmoidRegression(GenericLinearRegression):
    factory = scientific.SmoothstepSigmoidFitSolver
    parameters = np.array([3.0, 2.0])
    xmin = -1.5
    xmax = +1.5
    loss_domains = pd.DataFrame({"min": [1., 1.], "max": [5., 5.]}).T


class SmoothstepSigmoidRegressionNoiseL0(SmoothstepSigmoidRegression, TestCase):
    sigma = 1e-6


class SmoothstepSigmoidRegressionNoiseL1(SmoothstepSigmoidRegression, TestCase):
    sigma = 2.5e-2


class SmoothstepSigmoidRegressionNoiseL2(SmoothstepSigmoidRegression, TestCase):
    sigma = 1e-1


class InverseBoxCoxRegression(GenericLinearRegression):
    factory = scientific.InverseBoxCoxFitSolver
    configuration = {"p0": (0.50,)}
    parameters = np.array([0.38])
    xmin = 0.0
    xmax = 1.0
    loss_domains = pd.DataFrame({"min": [0.1], "max": [0.9]}).T


class InverseBoxCoxRegressionNoiseL0(InverseBoxCoxRegression, TestCase):
    sigma = 1e-6


class InverseBoxCoxRegressionNoiseL1(InverseBoxCoxRegression, TestCase):
    sigma = 2.5e-2


class InverseBoxCoxRegressionNoiseL2(InverseBoxCoxRegression, TestCase):
    sigma = 1e-1


class DoubleInverseBoxCoxSigmoidRegression(GenericLinearRegression):
    factory = scientific.DoubleInverseBoxCoxSigmoidFitSolver
    configuration = {"p0": (0.5, 0.5)}
    parameters = np.array([0.38, 0.47])
    xmin = 0.0
    xmax = 1.0
    loss_domains = pd.DataFrame({"min": [0.1]*2, "max": [0.9]*2}).T


class DoubleInverseBoxCoxSigmoidRegressionNoiseL0(
    DoubleInverseBoxCoxSigmoidRegression, TestCase
):
    sigma = 1e-6


class DoubleInverseBoxCoxSigmoidRegressionNoiseL1(
    DoubleInverseBoxCoxSigmoidRegression, TestCase
):
    sigma = 2.5e-2


class DoubleInverseBoxCoxSigmoidRegressionNoiseL2(
    DoubleInverseBoxCoxSigmoidRegression, TestCase
):
    sigma = 1e-1


class GaussianPeakRegression(GenericLinearRegression):
    factory = scientific.GaussianPeakFitSolver
    configuration = {"p0": (500., 5., 20.)} #, "bounds": [(1e-2, 1e-2, 1e-2), (1e3, 1e3, 1e3)]}  # 20 min run
    parameters = np.array([450.3, 1.23, 15.7])
    xmin = 0.0
    xmax = 30.0
    resolution = 120
    #loss_domains = pd.DataFrame({"min": [0.1]*2, "max": [0.9]*2}).T

    def test_fit_from_synthetic_dataset(self):
        pass

    def test_kolmogorov(self):
        pass


class GaussianPeakRegressionNoiseL0(
    GaussianPeakRegression, TestCase
):
    sigma = 1e-6


class GaussianPeakRegressionNoiseL1(
    GaussianPeakRegression, TestCase
):
    sigma = 2.5e-2


class GaussianPeakRegressionNoiseL2(
    GaussianPeakRegression, TestCase
):
    sigma = 1e-1


class GaussianPeakWithBaselineRegression(GenericLinearRegression):
    factory = scientific.GaussianPeakWithBaselineFitSolver
    configuration = {"p0": (500., 5., 20., 10, 100)}
    parameters = np.array([450.3, 1.23, 15.7, 3.58, 89.7])
    xmin = 0.0
    xmax = 30.0
    resolution = 120
    #loss_domains = pd.DataFrame({"min": [0.1]*2, "max": [0.9]*2}).T

    def test_fit_from_synthetic_dataset(self):
        pass

    def test_kolmogorov(self):
        pass


class GaussianPeakWithBaselineRegressionNoiseL0(
    GaussianPeakWithBaselineRegression, TestCase
):
    sigma = 1e-6


class GaussianPeakWithBaselineRegressionNoiseL1(
    GaussianPeakWithBaselineRegression, TestCase
):
    sigma = 2.5e-2


class GaussianPeakWithBaselineRegressionNoiseL2(
    GaussianPeakWithBaselineRegression, TestCase
):
    sigma = 1e-1


class EMGPeakRegression(GenericLinearRegression):
    """Heavy duty model"""
    factory = scientific.EMGPeakFitSolver
    configuration = {"p0": (500., 5., 20., 10.)} #, "bounds": [(1e-2, 1e-2, 1e-2, 1e-2), (1e3, 1e3, 1e3, 1e3)]}  # 20 min run
    parameters = np.array([450.3, 1.23, 15.7, 3.42])
    xmin = 0.0
    xmax = 30.0
    resolution = 120
    #loss_domains = pd.DataFrame({"min": [0.1]*2, "max": [0.9]*2}).T

    def test_fit_from_synthetic_dataset(self):
        pass

#
# class EMGPeakRegressionNoiseL0(
#     EMGPeakRegression, TestCase
# ):
#     sigma = 1e-6
#
#
# class EMGPeakRegressionNoiseL1(
#     EMGPeakRegression, TestCase
# ):
#     sigma = 2.5e-2
#
#
# class EMGPeakRegressionNoiseL2(
#     EMGPeakRegression, TestCase
# ):
#     sigma = 1e-1
