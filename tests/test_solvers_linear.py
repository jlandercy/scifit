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
        self.p = np.array([2, 3])
        self.x = np.linspace(0, 10, 21)
        self.solver = LinearFitSolver()
        self.y = self.solver.model(self.x, *self.p)

    def test_model_implementation(self):
        yhat = self.solver.model(self.x, *self.p)
        self.assertTrue(np.allclose(self.y, yhat))

    def test_model_fit(self):
        solution = self.solver.fit(self.x, self.y)
        self.assertIsInstance(solution, dict)
        self.assertSetEqual({"parameters", "covariance", "info", "message", "status"}, set(solution.keys()))
        self.assertTrue(np.allclose(solution["parameters"], self.p))
        for key in ["_xdata", "_ydata", "_solution", "_yhat", "_score"]:
            self.assertTrue(hasattr(self.solver, key))

