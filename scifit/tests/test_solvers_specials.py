"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`scifit.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

from scifit.solvers import specials
from scifit.tests.helpers import GenericTestFitSolver
from scifit.tests.test_solvers_linear import GenericLinearRegression


class DebyeRegression(GenericLinearRegression):
    factory = specials.DebyeFitSolver
    parameters = np.array([428.0])
    sigma = None
    xmin = 25.0
    xmax = 550.0


class DebyeRegressionNoiseL0(DebyeRegression, TestCase):
    sigma = 1e-6


class DebyeRegressionNoiseL1(DebyeRegression, TestCase):
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
    loss_resolution = 20


class CrankDiffusionRegressionNoiseL0(CrankDiffusionRegression, TestCase):
    sigma = 1e-6


# class CrankDiffusionRegressionNoiseL1(CrankDiffusionRegression, TestCase):
#     sigma = 5e-2
#
