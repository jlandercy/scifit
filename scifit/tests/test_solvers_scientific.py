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
    sigma = None


class GompertzRegressionNoiseL1(GompertzRegression, TestCase):
    sigma = 1e-3


class GompertzRegressionNoiseL2(GompertzRegression, TestCase):
    sigma = 1e-2


class GompertzRegressionNoiseL3(GompertzRegression, TestCase):
    sigma = 1e-1


class GompertzRegressionNoiseL4(GompertzRegression, TestCase):
    sigma = 1.0


# class GompertzRegressionNoiseL5(GompertzRegression, TestCase):
#     sigma = 10.


class LogGompertzRegression(GompertzRegression):
    factory = scientific.LogGompertzFitSolver


class LogGompertzRegressionNoiseL0(LogGompertzRegression, TestCase):
    sigma = None


class LogGompertzRegressionNoiseL1(LogGompertzRegression, TestCase):
    sigma = 1e-3


class LogGompertzRegressionNoiseL2(LogGompertzRegression, TestCase):
    sigma = 1e-2


class LogGompertzRegressionNoiseL3(LogGompertzRegression, TestCase):
    sigma = 1e-1


#
# class LogGompertzRegressionNoiseL4(LogGompertzRegression, TestCase):
#     sigma = 1.
#
#
# class LogGompertzRegressionNoiseL5(LogGompertzRegression, TestCase):
#     sigma = 10.


class GenericKineticRegression(GenericTestFitSolver):
    xmin = 1e-6
    xmax = 1e-3
    mode = "lin"
    sigma = None


class MichaelisMentenKineticRegression(GenericKineticRegression):
    factory = scientific.MichaelisMentenKineticFitSolver
    parameters = np.array([4.1e-3, 2.5e-5])


class MichaelisMentenKineticRegressionNoiseL0(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = None


class MichaelisMentenKineticRegressionNoiseL1(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 1e-3


class MichaelisMentenKineticRegressionNoiseL2(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 1e-2


class MichaelisMentenKineticRegressionNoiseL3(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 1e-1


class MichaelisMentenKineticRegressionNoiseL4(
    MichaelisMentenKineticRegression, TestCase
):
    sigma = 1.0


#
# class MichaelisMentenKineticRegressionNoiseL5(MichaelisMentenKineticRegression, TestCase):
#     sigma = 10.


class CooperativeHillKineticRegression(GenericKineticRegression):
    factory = scientific.HillKineticFitSolver
    parameters = np.array([1.12, 2.5e-1])


class CooperativeHillKineticRegressionNoiseL0(
    CooperativeHillKineticRegression, TestCase
):
    sigma = None


class CooperativeHillKineticRegressionNoiseL1(
    CooperativeHillKineticRegression, TestCase
):
    sigma = 1e-3


class CooperativeHillKineticRegressionNoiseL2(
    CooperativeHillKineticRegression, TestCase
):
    sigma = 1e-2


class CooperativeHillKineticRegressionNoiseL3(
    CooperativeHillKineticRegression, TestCase
):
    sigma = 1e-1


class CooperativeHillKineticRegressionNoiseL4(
    CooperativeHillKineticRegression, TestCase
):
    sigma = 1.0


class CooperativeHillKineticRegressionNoiseL5(
    CooperativeHillKineticRegression, TestCase
):
    sigma = 10.0


class CompetitiveHillKineticRegression(GenericKineticRegression):
    factory = scientific.HillKineticFitSolver
    parameters = np.array([0.72, 2.5e-1])


class CompetitiveHillKineticRegressionNoiseL0(
    CompetitiveHillKineticRegression, TestCase
):
    sigma = None


class CompetitiveHillKineticRegressionNoiseL1(
    CompetitiveHillKineticRegression, TestCase
):
    sigma = 1e-3


class CompetitiveHillKineticRegressionNoiseL2(
    CompetitiveHillKineticRegression, TestCase
):
    sigma = 1e-2


class CompetitiveHillKineticRegressionNoiseL3(
    CompetitiveHillKineticRegression, TestCase
):
    sigma = 1e-1


class CompetitiveHillKineticRegressionNoiseL4(
    CompetitiveHillKineticRegression, TestCase
):
    sigma = 1.0


class CompetitiveHillKineticRegressionNoiseL5(
    CompetitiveHillKineticRegression, TestCase
):
    sigma = 10.0
