"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`scifit.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

from scifit.solvers.linear import LinearFitSolver


class GenericTestFitSolver:

    seed = 1234567890
    factory = LinearFitSolver
    kwargs = {}
    p = np.array([2, 3])
    x = np.linspace(0, 10, 21).reshape(-1, 1)
    s = 0.

    def setUp(self) -> None:
        np.random.seed(self.seed)
        self.solver = self.factory(**self.kwargs)
        self.y = self.solver.model(self.x, *self.p) + self.s*np.random.rand(self.x.shape[0])

    def test_signature(self):
        s = self.solver.signature
        n = self.solver.parameter_space_size
        self.assertEqual(len(s.parameters) - 1, n)

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
                    solution["parameters"][i],
                    atol=10*np.sqrt(solution["covariance"][i][i])
                )
            )

    def test_plot_fit(self):
        name = self.__class__.__name__
        title = "{} (seed={})".format(name, self.seed)
        self.solver.fit(self.x, self.y)
        for i, axe in enumerate(self.solver.plot_fit(title=title)):
            axe.figure.savefig("media/{}_fit_x{}.png".format(name, i))
            plt.close(axe.figure)


class TestPerfectLinearRegression(GenericTestFitSolver, TestCase):
    s = 0.


class TestSmallNoisyLinearRegression(GenericTestFitSolver, TestCase):
    s = 1e-3


class TestMediumNoisyLinearRegression(GenericTestFitSolver, TestCase):
    s = 0.25


class TestLargeNoisyLinearRegression(GenericTestFitSolver, TestCase):
    s = 1.


class TestVeryLargeNoisyLinearRegression(GenericTestFitSolver, TestCase):
    s = 25.


class TestExtraLargeNoisyLinearRegression(GenericTestFitSolver, TestCase):
    s = 1e3
