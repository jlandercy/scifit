"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`scifit.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

from scifit.solvers.linear import *


class GenericTestFitSolver:

    seed = 789
    factory = LinearFitSolver
    kwargs = {}
    p = np.array([2., 3.])
    x = np.linspace(-1., 1., 30).reshape(-1, 1)
    s = 0.

    def setUp(self) -> None:
        np.random.seed(self.seed)
        self.solver = self.factory(**self.kwargs)
        self.y = self.solver.model(self.x, *self.p)
        self.e = self.s*np.abs(self.y)*np.random.uniform(low=-.5, high=.5, size=self.x.shape[0])
        self.y += self.e

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

    def test_parameters_domain_linear_auto(self):
        solution = self.solver.fit(self.x, self.y)
        domains = self.solver.parameter_domains()

    def test_parameters_domain_linear_fixed(self):
        solution = self.solver.fit(self.x, self.y)
        domains = self.solver.parameter_domains(xmax=100.)

    def test_parameters_domain_logarithmic_auto(self):
        solution = self.solver.fit(self.x, self.y)
        domains = self.solver.parameter_domains(mode="log")

    def test_parameters_domain_logarithmic_fixed(self):
        solution = self.solver.fit(self.x, self.y)
        domains = self.solver.parameter_domains(mode="log", xmax=100.)

    def test_plot_fit(self):
        name = self.__class__.__name__
        title = "{} (seed={:d})".format(name, self.seed)
        self.solver.fit(self.x, self.y)
        for i, axe in enumerate(self.solver.plot_fit(title=title)):
            axe.figure.savefig("media/{}_fit_x{}.png".format(name, i))
            plt.close(axe.figure)

    def test_plot_score(self):
        name = self.__class__.__name__
        title = "{} (seed={:d})".format(name, self.seed)
        self.solver.fit(self.x, self.y)
        for i, axe in enumerate(self.solver.plot_score(title=title)):
            axe.figure.savefig("media/{}_score_b{}_b{}.png".format(name, *axe._pair_indices))
            plt.close(axe.figure)


class GenericConstantRegression(GenericTestFitSolver):
    factory = ConstantFitSolver
    p = np.array([5.])


class ConstantRegressionNoiseL0(GenericConstantRegression, TestCase):
    s = 0.


class ConstantRegressionNoiseL1(GenericConstantRegression, TestCase):
    s = 1.e-3


class ConstantRegressionNoiseL2(GenericConstantRegression, TestCase):
    s = 1.e-2


class ConstantRegressionNoiseL3(GenericConstantRegression, TestCase):
    s = 1.e-1


class ConstantRegressionNoiseL4(GenericConstantRegression, TestCase):
    s = 1.


class ConstantRegressionNoiseL5(GenericConstantRegression, TestCase):
    s = 10.


class GenericProportionalRegression(GenericTestFitSolver):
    factory = ProportionalFitSolver
    p = np.array([5.])


class ProportionalRegressionNoiseL0(GenericProportionalRegression, TestCase):
    s = 0.


class ProportionalRegressionNoiseL1(GenericProportionalRegression, TestCase):
    s = 1.e-3


class ProportionalRegressionNoiseL2(GenericProportionalRegression, TestCase):
    s = 1.e-2


class ProportionalRegressionNoiseL3(GenericProportionalRegression, TestCase):
    s = 1.e-1


class ProportionalRegressionNoiseL4(GenericProportionalRegression, TestCase):
    s = 1.


class ProportionalRegressionNoiseL5(GenericProportionalRegression, TestCase):
    s = 10.


class GenericLinearRegression(GenericTestFitSolver):
    factory = LinearFitSolver
    p = np.array([2., 3.])


class LinearRegressionNoiseL0(GenericLinearRegression, TestCase):
    s = 0.


class LinearRegressionNoiseL1(GenericLinearRegression, TestCase):
    s = 1.e-3


class LinearRegressionNoiseL2(GenericLinearRegression, TestCase):
    s = 1.e-2


class LinearRegressionNoiseL3(GenericLinearRegression, TestCase):
    s = 1.e-1


class LinearRegressionNoiseL4(GenericLinearRegression, TestCase):
    s = 1.


class LinearRegressionNoiseL5(GenericLinearRegression, TestCase):
    s = 10.


class GenericParabolicRegression(GenericTestFitSolver):
    factory = ParabolicFitSolver
    p = np.array([1., 2., 3.])


class ParabolicRegressionNoiseL0(GenericParabolicRegression, TestCase):
    s = 0.


class ParabolicRegressionNoiseL1(GenericParabolicRegression, TestCase):
    s = 1.e-3


class ParabolicRegressionNoiseL2(GenericParabolicRegression, TestCase):
    s = 1.e-2


class ParabolicRegressionNoiseL3(GenericParabolicRegression, TestCase):
    s = 1.e-1


class ParabolicRegressionNoiseL4(GenericParabolicRegression, TestCase):
    s = 1.


class ParabolicRegressionNoiseL5(GenericParabolicRegression, TestCase):
    s = 10.


class GenericCubicRegression(GenericTestFitSolver):
    factory = CubicFitSolver
    p = np.array([1., 2., 3., 4.])


class CubicRegressionNoiseL0(GenericCubicRegression, TestCase):
    s = 0.


class CubicRegressionNoiseL1(GenericCubicRegression, TestCase):
    s = 1.e-3


class CubicRegressionNoiseL2(GenericCubicRegression, TestCase):
    s = 1.e-2


class CubicRegressionNoiseL3(GenericCubicRegression, TestCase):
    s = 1.e-1


class CubicRegressionNoiseL4(GenericCubicRegression, TestCase):
    s = 1.


class CubicRegressionNoiseL5(GenericCubicRegression, TestCase):
    s = 10.


class GenericLinearRootRegression(GenericTestFitSolver):
    factory = LinearRootFitSolver
    p = np.array([1., 2., 3.])


class LinearRootRegressionNoiseL0(GenericLinearRootRegression, TestCase):
    s = 0.


class LinearRootRegressionNoiseL1(GenericLinearRootRegression, TestCase):
    s = 1.e-3


class LinearRootRegressionNoiseL2(GenericLinearRootRegression, TestCase):
    s = 1.e-2


class LinearRootRegressionNoiseL3(GenericLinearRootRegression, TestCase):
    s = 1.e-1


class LinearRootRegressionNoiseL4(GenericLinearRootRegression, TestCase):
    s = 1.


class LinearRootRegressionNoiseL5(GenericLinearRootRegression, TestCase):
    s = 10.

