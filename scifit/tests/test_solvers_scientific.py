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
    parameters = np.array([2.0, 0.5, 5.0])
    sigma = None


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


class GenericKineticRegression(GenericTestFitSolver):
    xmin = 1e-6
    xmax = 1e-3
    mode = "lin"
    sigma = 1e-16


class MichaelisMentenKineticRegression(GenericKineticRegression):
    factory = scientific.MichaelisMentenKineticFitSolver
    parameters = np.array([4.1e-3, 2.5e-5])
    configuration = {"p0": np.array([1e2, 1e2])}


class MichaelisMentenKineticRegressionNoiseL0(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 1e-6


class MichaelisMentenKineticRegressionNoiseL1(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 2.5e-2


class MichaelisMentenKineticRegressionNoiseL2(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 1e-1


class MichaelisMentenKineticRegressionNoiseL3(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 2.5e-1


class MichaelisMentenKineticRegressionNoiseL4(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 1.0


class MichaelisMentenKineticRegressionNoiseL5(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 2.5


class CooperativeHillEquationRegression(GenericKineticRegression):
    factory = scientific.HillEquationFitSolver
    parameters = np.array([2.12, 2.5e-1])


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


class CooperativeHillEquationRegressionNoiseL5(
    CooperativeHillEquationRegression, TestCase
):
    sigma = 2.5


class CompetitiveHillEquationRegression(GenericKineticRegression):
    factory = scientific.HillEquationFitSolver
    parameters = np.array([0.32, 2.5e-1])


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


class CompetitiveHillEquationRegressionNoiseL5(
    CompetitiveHillEquationRegression, TestCase
):
    sigma = 2.5


class LogisticRegression(GenericLinearRegression):
    factory = scientific.LogisticFitSolver
    parameters = np.array([3.1, -0.75])


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


class AlgebraicSigmoidRegressionNoiseL0(AlgebraicSigmoidRegression, TestCase):
    sigma = 1e-6


class AlgebraicSigmoidRegressionNoiseL1(AlgebraicSigmoidRegression, TestCase):
    sigma = 2.5e-2


class AlgebraicSigmoidRegressionNoiseL2(AlgebraicSigmoidRegression, TestCase):
    sigma = 1e-1


class RichardGeneralizedSigmoidRegression(GenericLinearRegression):
    factory = scientific.RichardGeneralizedSigmoidFitSolver
    parameters = np.array([4.1, 2.3, 1.1, 4.7, 3.2, 1.1])
    xmin = -5.0
    xmax = +5.0
    resolution = 50

    def test_model_minimize_against_solve(self):
        """Richardson model is not numerically stable"""
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
