"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`newproject.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

from scifit.solvers.linear import LinearFitSolver


class GenericTestFitSolver:

    factory = LinearFitSolver
    kwargs = {}
    p = np.array([2, 3])
    x = np.linspace(0, 10, 21)
    s = 0.

    def setUp(self) -> None:
        self.solver = self.factory(**self.kwargs)
        self.y = self.solver.model(self.x, *self.p) + self.s*np.random.rand(self.x.shape[0])

    def test_model_implementation(self):
        yhat = self.solver.model(self.x, *self.p)
        self.assertTrue(np.allclose(self.y, yhat, rtol=10*self.s))

    def test_model_fit_signature(self):
        solution = self.solver.fit(self.x, self.y)
        self.assertIsInstance(solution, dict)
        self.assertSetEqual({"parameters", "covariance", "info", "message", "status"}, set(solution.keys()))
        for key in ["_xdata", "_ydata", "_solution", "_yhat", "_score"]:
            self.assertTrue(hasattr(self.solver, key))

    def test_model_fit_parameters(self):
        solution = self.solver.fit(self.x, self.y)
        for i in range(self.p.shape[0]):
            self.assertTrue(
                np.allclose(
                    self.p[i],
                    self.solver._solution["parameters"][i],
                    atol=10*np.sqrt(self.solver._solution["covariance"][i][i])
                )
            )


class TestPerfectLinearRegression(GenericTestFitSolver, TestCase):
    pass


class TestSmallNoisyLinearRegression(GenericTestFitSolver, TestCase):
    s = 1e-3


class TestMediumNoisyLinearRegression(GenericTestFitSolver, TestCase):
    s = 0.25


class TestLargeNoisyLinearRegression(GenericTestFitSolver, TestCase):
    s = 1.
