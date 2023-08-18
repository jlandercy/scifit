"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`scifit.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

from scifit.solvers import specials
from scifit.tests.helpers import GenericTestFitSolver
from scifit.tests.test_solvers_linear import GenericLinearRegression


class DebyeInternalEnergyRegression(GenericLinearRegression):
    factory = specials.DebyeInternalEnergyFitSolver
    parameters = np.array([428.0])
    sigma = None
    xmin = 25.0
    xmax = 550.0


class DebyeInternalEnergyRegressionNoiseL0(DebyeInternalEnergyRegression, TestCase):
    sigma = 1e-6


class DebyeInternalEnergyRegressionNoiseL1(DebyeInternalEnergyRegression, TestCase):
    sigma = 5e-2


class CrankDiffusionRegression(GenericLinearRegression):
    factory = specials.CrankDiffusionFitSolver
    parameters = np.array([3.9, 2e-11])
    configuration = {"p0": np.array([5.0, 1e-10])}
    sigma = None
    xmin = 1e0
    xmax = 1e6
    resolution = 20
    mode = "log"
    log_x = True
    log_y = True
    log_loss = True
    loss_resolution = 200  # Heavy on CPU but space is large

#
# class CrankDiffusionRegressionNoiseL0(CrankDiffusionRegression, TestCase):
#     sigma = 1e-6


class CrankDiffusionRegressionNoiseL1(CrankDiffusionRegression, TestCase):
    sigma = 5e-2


class RaneyKetonDehydrogenationRegression(GenericLinearRegression):
    factory = specials.RaneyKetonDehydrogenationFitSolver
    parameters = np.array([4.75360716e-02, 7.80055896e+01])
    configuration = {"p0": np.array([1e-2, 1e2])}
    sigma = None
    xmin = 0.1
    xmax = 1e4


# class RaneyKetonDehydrogenationRegressionNoiseL0(RaneyKetonDehydrogenationRegression, TestCase):
#     sigma = 1e-6
#
#
# class RaneyKetonDehydrogenationRegressionNoiseL1(RaneyKetonDehydrogenationRegression, TestCase):
#     sigma = 5e-2
