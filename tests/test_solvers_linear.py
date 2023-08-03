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

    seed = 1234567890
    factory = LinearFitSolver
    kwargs = {}
    p = np.array([2., 3.])
    x = np.linspace(-1., 1., 30).reshape(-1, 1)
    s = 0.

    def setUp(self) -> None:
        np.random.seed(self.seed)
        self.solver = self.factory(**self.kwargs)
        self.y = self.solver.model(self.x, *self.p)
        self.y += (self.s/10)*np.abs(self.y)*np.random.uniform(low=-.5, high=.5, size=self.x.shape[0])

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
    seed = 1234567890
    factory = ConstantFitSolver
    kwargs = {}
    p = np.array([10.])
    s = 0.


class ConstantRegressionNoiseL0(GenericConstantRegression, TestCase):
    s = 0.


class ConstantRegressionNoiseL1(GenericConstantRegression, TestCase):
    s = 1.e-2


class ConstantRegressionNoiseL2(GenericConstantRegression, TestCase):
    s = 1.e-1


class ConstantRegressionNoiseL3(GenericConstantRegression, TestCase):
    s = 1.


class ConstantRegressionNoiseL4(GenericConstantRegression, TestCase):
    s = 10.


class ConstantRegressionNoiseL5(GenericConstantRegression, TestCase):
    s = 100.


class GenericProportionalRegression(GenericTestFitSolver):
    seed = 1234567890
    factory = ProportionalFitSolver
    kwargs = {}
    p = np.array([5.])
    s = 0.


class ProportionalRegressionNoiseL0(GenericProportionalRegression, TestCase):
    s = 0.


class ProportionalRegressionNoiseL1(GenericProportionalRegression, TestCase):
    s = 1.e-2


class ProportionalRegressionNoiseL2(GenericProportionalRegression, TestCase):
    s = 1.e-1


class ProportionalRegressionNoiseL3(GenericProportionalRegression, TestCase):
    s = 1.


class ProportionalRegressionNoiseL4(GenericProportionalRegression, TestCase):
    s = 10.


class ProportionalRegressionNoiseL5(GenericProportionalRegression, TestCase):
    s = 100.


class GenericLinearRegression(GenericTestFitSolver):
    seed = 1234567890
    factory = LinearFitSolver
    kwargs = {}
    p = np.array([2., 3.])
    s = 0.


class LinearRegressionNoiseL0(GenericLinearRegression, TestCase):
    s = 0.


class LinearRegressionNoiseL1(GenericLinearRegression, TestCase):
    s = 1.e-2


class LinearRegressionNoiseL2(GenericLinearRegression, TestCase):
    s = 1.e-1


class LinearRegressionNoiseL3(GenericLinearRegression, TestCase):
    s = 1.


class LinearRegressionNoiseL4(GenericLinearRegression, TestCase):
    s = 10.


class LinearRegressionNoiseL5(GenericLinearRegression, TestCase):
    s = 100.


class GenericParabolaRegression(GenericTestFitSolver):
    seed = 1234567890
    factory = ParabolaFitSolver
    kwargs = {}
    p = np.array([1., -2., 3.])
    s = 0.


class ParabolaRegressionNoiseL0(GenericParabolaRegression, TestCase):
    s = 0.


class ParabolaRegressionNoiseL1(GenericParabolaRegression, TestCase):
    s = 1.e-2


class ParabolaRegressionNoiseL2(GenericParabolaRegression, TestCase):
    s = 1.e-1


class ParabolaRegressionNoiseL3(GenericParabolaRegression, TestCase):
    s = 1.


class ParabolaRegressionNoiseL4(GenericParabolaRegression, TestCase):
    s = 10.


class ParabolaRegressionNoiseL5(GenericParabolaRegression, TestCase):
    s = 100.
