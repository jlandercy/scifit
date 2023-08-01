"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`newproject.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

from scifit.solvers.linear import LinearFitSolver
from scifit.errors.base import *


class TestLinearFitSolver(TestCase):

    def setUp(self) -> None:
        self.x = np.linspace(0, 10, 21)
        self.y = 2*self.x + 3
        self.solver = LinearFitSolver(self.x, self.y)

    def test_model_implementation(self):
        yhat = self.solver.model(self.x, 2, 3)
        self.assertTrue(np.allclose(self.y, yhat))

    def test_model_fit(self):
        solution = self.solver.solve()
        self.assertIsInstance(solution, dict)
        self.assertSetEqual({"parameters", "covariance", "info", "message", "status"}, set(solution.keys()))
