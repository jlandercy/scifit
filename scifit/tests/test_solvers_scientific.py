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
    parameters = np.array([3.1, 10.27])


class LogisticRegressionNoiseL0(
    LogisticRegression, TestCase
):
    sigma = 1e-6
